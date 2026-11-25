"""Run a full-update smoke before one G160; require idle GPUs at launch; later arrivals do not preempt this run."""
from pathlib import Path
import argparse,fcntl,json,os,signal,subprocess,sys,time

def check_idle(uuids):
    # Query only selected devices; a broken unrelated GPU can break nvidia-smi queries.
    import pynvml as nv
    nv.nvmlInit()
    try:
        for uid in uuids:
            handle=nv.nvmlDeviceGetHandleByUUID(uid)
            if nv.nvmlDeviceGetComputeRunningProcesses(handle):
                raise RuntimeError(f'Selected GPU occupied: {uid}; not launching')
            memory=nv.nvmlDeviceGetMemoryInfo(handle)
            utilization=nv.nvmlDeviceGetUtilizationRates(handle).gpu
            if memory.free<=44000*1024**2 or utilization>5:
                raise RuntimeError(f'Selected GPU not idle: {uid}')
    finally:nv.nvmlShutdown()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',default='formal_g160')
    parser.add_argument('--reuse-smoke',action='store_true')
    a=parser.parse_args();r=a.root
    if (r/'runs'/a.run).exists():raise FileExistsError('Use a fresh formal run name')
    cfg=json.loads((r/'config.json').read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(cfg['gpu_uuids']),NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',
             OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1',
             PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True',TMPDIR=str(r/'tmp'),MPLCONFIGDIR=str(r/'cache/matplotlib'))
    (r/'tmp').mkdir(exist_ok=True);(r/'logs').mkdir(exist_ok=True)
    def launch(name,micro,updates,smoke):
        if (r/'STOP_REQUEST').exists():raise RuntimeError('Stop requested')
        check_idle(cfg['gpu_uuids'])
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone',f"--nproc_per_node={len(cfg['gpu_uuids'])}",str(r/'code/train_matched.py'),
             '--root',str(r),'--run',name,'--micro',str(micro),'--updates',str(updates)]+(['--smoke'] if smoke else [])
        logpath=r/'logs'/f'{name}.log'
        with logpath.open('w') as log:
            p=subprocess.Popen(cmd,env=env,cwd=r/'code',stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('START',name,p.pid,flush=True)
            try:
                while p.poll() is None:
                    # Launch-time exclusivity only: a later process is not a stop reason.
                    if (r/'STOP_REQUEST').exists():raise RuntimeError('Stop requested')
                    time.sleep(5)
            except BaseException:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
                raise
        return p.returncode,logpath
    with (r/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        selected=None
        if a.reuse_smoke:
            proof=json.loads((r/'smoke_passed.json').read_text())
            assert proof['result']['strict_load'] and proof['result']['finite']
            assert (proof['result']['G'],proof['result']['F'])==(4,20)
            selected=proof['micro']
            print('REUSE_VALIDATED_SMOKE',selected,flush=True)
        for micro in (() if a.reuse_smoke else tuple(cfg['smoke_micro_candidates'])):
            name=f'smoke_micro{micro}'
            rc,path=launch(name,micro,4,True)
            if rc:
                if 'out of memory' not in path.read_text(errors='replace').lower():raise RuntimeError(f'Smoke failed; inspect {path}')
                print('OOM_REDUCE_MICRO',micro,flush=True);time.sleep(10);continue
            result=json.loads((r/'runs'/name/'complete.json').read_text())
            assert result['G']==4 and result['F']==20 and result['strict_load'] and result['finite']
            for rank in range(len(cfg['gpu_uuids'])):
                rows=[json.loads(x) for x in (r/'runs'/name/f'rank{rank}.jsonl').read_text().splitlines()]
                assert len(rows)==20 and sum(x['student_updated'] for x in rows)==4
                assert [x['depth'] for x in rows if x['student_updated']]==[0,1,2,3]
                assert len(json.loads((r/'runs'/name/'eval/g000004'/f'rank{rank}.json').read_text())['samples'])==8//len(cfg['gpu_uuids'])
            for rank in range(len(cfg['gpu_uuids'])):
                precision=json.loads((r/'runs'/name/f'precision_rank{rank}.json').read_text())
                assert precision['fake:autocast=False:grad=False']['output']=='torch.float32'
                assert precision['student:autocast=False:grad=False']['output']=='torch.float32'
                assert precision['teacher:autocast=False:grad=False']['weight']=='torch.bfloat16'
                rows=[json.loads(x) for x in (r/'runs'/name/f'rank{rank}.jsonl').read_text().splitlines()]
                assert rows[0]['student_lr']==0 and rows[5]['student_lr']==cfg['student_lr']*.64
                assert rows[8]['fake_lr']==cfg['fake_lr']
            peak=max(json.loads(line)['peak_GiB'] for rank in range(len(cfg['gpu_uuids']))
                     for line in (r/'runs'/name/f'rank{rank}.jsonl').read_text().splitlines())
            if peak>cfg['smoke_max_allocated_GiB']:
                (r/'runs'/name/'student.pt').unlink()
                print('INSUFFICIENT_HEADROOM_REDUCE_MICRO',micro,peak,flush=True)
                continue
            selected=micro
            (r/'smoke_passed.json').write_text(json.dumps(dict(micro=micro,accumulation=64//(len(cfg['gpu_uuids'])*micro),result=result),indent=2))
            # Smoke state was loaded successfully; retain logs/config/images only.
            (r/'runs'/name/'student.pt').unlink()
            print('SMOKE_PASSED',micro,flush=True);time.sleep(5);break
        if selected is None:raise RuntimeError('No microbatch fits; do not launch formal training')
        rc,path=launch(a.run,selected,160,False)
        if rc:raise RuntimeError(f'Formal training failed; inspect {path}')
        done=json.loads((r/'runs'/a.run/'complete.json').read_text());assert (done['G'],done['F'])==(160,800)
        (r/'complete.json').write_text(json.dumps(done,indent=2));print('COMPLETE_G160_F800',flush=True)
        # Reuse the already validated, resumable fixed COCO10K/CLIP1024 evaluator.
        # Its own launch checks must pass again after training releases the GPUs.
        print('START_COCO10K_COMPARISON',flush=True)
        subprocess.run([sys.executable,'-u',str(r/'coco10k/code/run_queue.py'),str(r/'coco10k')],check=True)
if __name__=='__main__':main()
