"""Local A800 pipeline: after the uniform control, COCO10K B/C, then one winner-based G160."""
import fcntl,hashlib,json,math,os,shutil,subprocess,sys,tarfile,time
from pathlib import Path
import numpy as np
from launch import command,gpu_check
from run_quota_bc import verify
from turning_batch64_workflow import descendants,foreign_on,stop_own

root=Path(os.environ['SDXL_ROOT'])
work=Path(os.environ['SDXL_WORK'])
inputs=work/'bc_texture_inputs'
previous=root/'outputs/turning_BC_paired_quota_20261102'
job=root/'outputs/bc_winner_late_mix_20261102'
evalout=job/'coco10k'
code=Path(__file__).resolve().parent
public_hash='8f045bfbc2507ee74547665dadeb0eb87b33cfdc8805c83275089d926c5d2246'
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',
         OPENBLAS_NUM_THREADS='4',TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False));tmp.replace(path)

def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def check_stop():
    if any((d/name).exists() for d in (job,previous) for name in ('USER_STOP','STOP_REQUEST')):
        raise RuntimeError('User stop requested')

def prepare():
    check_stop()
    # Hashes also keep partially uploaded reference archives out of extraction.
    archive=root/'tmp/bc_public_reference.tar'
    while not (inputs/'public_inputs_ready.json').exists() or not archive.exists() or archive.stat().st_size!=archive_size:
        check_stop();time.sleep(20)
    if sha(archive)!=public_hash:raise ValueError('Reference archive SHA mismatch')
    with tarfile.open(archive) as t:t.extractall(inputs,filter='data')
    assert sha(inputs/'models/metrics/clean_fid/inception-2015-12-05.pt')=='f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'
    reference=inputs/'reference_features/real'
    ids=[]
    for file in sorted(reference.glob('*.npz')):
        with np.load(file) as z:
            ii=z['indices'].tolist();ids.extend(ii)
            assert all(z[k].shape==(len(ii),2048) and np.isfinite(z[k]).all() for k in ('full','patch'))
    assert ids==list(range(10000))
    features=evalout/'features';features.mkdir(parents=True,exist_ok=True)
    link=features/'real'
    if not link.exists():link.symlink_to(reference,target_is_directory=True)
    assert link.resolve()==reference.resolve()
    manifest=json.loads((code/'coco_reference_manifest.json').read_text())
    assert manifest['n']==10000 and len(manifest['samples'])==10000
    assert manifest['csv_sha256']=='fc1efb7dd0f7d561a999125340fc4b10632d28f2547c17b174170514889c8599'
    assert all(r['index']==i and r['seed']==20261020+i for i,r in enumerate(manifest['samples']))
    models={}
    for arm in ('B','C'):
        dest=Path(verify(previous/arm))
        cfg=json.loads((previous/arm/'config.json').read_text())
        assert cfg['schedule']==[999,749,499,249,-1]
        models[arm]=dict(checkpoint=str(dest),world_size=1,student_updates=160,config=cfg,
                         shards=[dict(name='student_rank0.pt',sha256=sha(dest/'student_rank0.pt'))])
    manifest['models']=models
    path=evalout/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:save(path,manifest)
    print('INPUTS_VERIFIED',flush=True)

def on_idle_gpu(cmd,log):
    while True:
        check_stop()
        try:cards=gpu_check([0],76);break
        except RuntimeError:time.sleep(30)
    with log.open('a') as f:
        p=subprocess.Popen(cmd,cwd=code,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print('STARTED',log.name,p.pid,flush=True)
        try:
            while p.poll() is None:
                check_stop()
                if foreign_on({cards[0][1]},descendants(p.pid)):raise RuntimeError('GPU resource conflict')
                time.sleep(10)
            if p.returncode:raise RuntimeError(f'{log} exited {p.returncode}')
        except BaseException:stop_own(p);raise

def main():
    job.mkdir(exist_ok=True)
    lock=(job/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX if '--wait-lock' in sys.argv else fcntl.LOCK_EX|fcntl.LOCK_NB)
    prepare()
    print('WAITING_FOR_UNIFORM_G160',flush=True)
    gate=(previous/'uniform_queue.lock').open('a');fcntl.flock(gate,fcntl.LOCK_EX)
    check_stop()
    assert (previous/'uniform_control_complete.json').is_file()
    verify(previous/'uniform_control')
    base=[sys.executable,'-u',str(code/'evaluate.py')]
    opts=['--root',str(inputs),'--out',str(evalout)]
    if not (evalout/'complete.json').exists():
        on_idle_gpu(base+['worker',*opts,'--methods','B,C','--batch','4'],evalout/'worker.log')
        with (evalout/'metrics.log').open('a') as f:
            subprocess.run(base+['metrics',*opts],cwd=code,env={**env,'CUDA_VISIBLE_DEVICES':''},stdout=f,stderr=subprocess.STDOUT,check=True)
    scores=json.loads((evalout/'complete.json').read_text())['scores']
    assert set(scores)=={'B','C'}
    assert all(s['n']==10000 and all(math.isfinite(s[k]) for k in ('FID','Patch_FID')) for s in scores.values())
    winner=min(scores,key=lambda a:(scores[a]['Patch_FID'],scores[a]['FID'],a))
    selection=dict(scores=scores,winner=winner,rule='lowest native center299 Patch_FID; tie: lowest FID, then arm name')
    save(job/'selection.json',selection);print('WINNER',json.dumps(selection),flush=True)
    from control_after_bc import run_control
    run_control()
    cfgfile=root/'code/sdxl_research/configs'/f'turning_{winner}_quota_a800_g160.json'
    cfg=json.loads(cfgfile.read_text())
    cfg.update(fake_track_pilot=True,fake_student_mix=.01,fake_mix_start_g=1)
    save(job/'effective_config.json',cfg)
    path=job/'late_mix'
    if not path.exists():
        if shutil.disk_usage(job).free<12*1024**3:raise RuntimeError('Insufficient space for final weights')
        on_idle_gpu(command(cfg,root,f'{job.name}/late_mix',1),job/'late_mix.log')
    dest=verify(path)
    logs=[json.loads(l) for l in (path/'rank0.jsonl').read_text().splitlines()]
    checks=[r for r in logs if r['phase']=='heldout_fake_tracking']
    assert [r['g'] for r in checks]==sorted([1,*range(8,161,8),121])
    for r in checks:
        assert r['student_mix']==.01
        assert [z['time'] for z in r['results']]==[60,499,940]
        assert all(math.isfinite(v) for z in r['results'] for v in z.values())
    save(job/'training_complete.json',dict(winner=winner,final_weights=dest,heldout_checks=checks))
    print('WINNER_G160_COMPLETE',flush=True)

if __name__=='__main__':
    archive_size=int((code/'reference_size.txt').read_text())
    main()
