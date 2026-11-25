"""Run on the storage/evaluation node; reuse its previously scored COCO protocol."""
import fcntl,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/data/shenglin.zeng/a50k_20261123/eval_A_milestones_20261125')
old=Path('/data3/shenglin.zeng/new_models_eval_20261114')
steps={20000:313,30000:470,40000:627,50000:784}
def read(p):return json.loads(p.read_text())
def free():
    def smi(q):return subprocess.check_output(['nvidia-smi',q,'--format=csv,noheader,nounits'],text=True).splitlines()
    busy={s.split(',')[0].strip() for s in smi('--query-compute-apps=gpu_uuid,pid')}
    return [r[1] for line in smi('--query-gpu=index,uuid,memory.free,utilization.gpu')
            for r in [[t.strip() for t in line.split(',')]] if r[1] not in busy and int(r[2])>21000 and int(r[3])<=5]
def main():
    lock=(root/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for k in ('models','selection.json'):
        if not (root/k).exists():(root/k).symlink_to(old/k)
    ref=read(old/'results/A_terminal_only/manifest.json')
    reference=read(root.parent/'datasets/coco10k_reference.json')
    for k in ('samples','visual_indices','csv_sha256'):assert ref[k]==reference[k]
    assert len(ref['samples'])==10000 and len(ref['visual_indices'])==128
    jobs={}
    for samples,g in steps.items():
        arm=f'A{samples}';out=root/'results'/arm;out.mkdir(exist_ok=True,parents=True)
        ck=root.parent/'checkpoints/singlepass'/f'samples{samples:05d}_g{g:06d}'
        meta=read(ck/'complete.json');cfg=meta['config'];pr=meta['progress']
        assert pr==dict(gcount=g,fcount=2*g,dcount=0,student_samples=samples)
        assert meta['world_size']==4 and cfg['schedule']==[999,749,499,249,-1]
        manifest=dict(reference,models={arm:dict(checkpoint=str(ck),world_size=4,config=cfg)})
        (out/'manifest.json').write_text(json.dumps(manifest))
        (out/'features').mkdir(exist_ok=True)
        link=out/'features/real'
        if not link.exists():link.symlink_to(old/'reference/features/real')
        if not (out/'all_complete.json').exists():jobs[arm]=None
    while jobs:
        reserved=set()
        for arm,entry in list(jobs.items()):
            if entry is None:continue
            proc,uuid,log=entry
            rc=proc.poll()
            if rc is None:reserved.add(uuid);continue
            log.close()
            if rc:raise RuntimeError(f'{arm} exited {rc}')
            assert (root/'results'/arm/'all_complete.json').is_file()
            del jobs[arm]
        available=[u for u in free() if u not in reserved]
        for arm,entry in list(jobs.items()):
            if entry is not None or not available:continue
            uuid=available.pop(0)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
            log=(root/'results'/arm/'worker.log').open('a')
            proc=subprocess.Popen([sys.executable,'-u',str(root/'code/worker.py'),str(root),arm],env=env,stdout=log,stderr=subprocess.STDOUT)
            jobs[arm]=proc,uuid,log;print('START',arm,uuid,proc.pid,flush=True)
        if jobs:time.sleep(15)
    result={}
    for n in steps:
        arm=f'A{n}';out=root/'results'/arm
        score=read(out/'all_complete.json')
        import math
        assert score['fid_n']==10000 and score['clip_n']==1024 and all(math.isfinite(score[k]) for k in ['FID','Patch_FID','CLIP'])
        assert {p.stem for p in (out/'images'/arm).glob('*.png')}=={f'{i:05d}' for i in reference['visual_indices']}
        result[arm]=score
    tmp=root/'summary.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(root/'summary.json')
if __name__=='__main__':main()
