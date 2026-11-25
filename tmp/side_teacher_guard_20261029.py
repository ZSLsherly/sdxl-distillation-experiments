import os,sys,subprocess,time,csv,io,signal,json
from pathlib import Path
root=Path('/data1/shenglin.zeng/teacher_turning_20261026');gpus='1,4'
run=root/'outputs/side_teacher_perf_20261029';run.mkdir(exist_ok=False)
def apps():
    rows=csv.reader(io.StringIO(subprocess.check_output(['nvidia-smi','--id='+gpus,'--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)))
    return {int(r[1]) for r in rows if len(r)>1}
def descendants(pid):
    ids={pid}
    try:
        for ch in (Path('/proc')/str(pid)/'task'/str(pid)/'children').read_text().split():ids.update(descendants(int(ch)))
    except FileNotFoundError:pass
    return ids
for setting in ('1','0'):
    if apps():raise RuntimeError('Selected GPUs occupied; not launching')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpus,NCCL_P2P_DISABLE=setting,NCCL_IB_DISABLE='1',OMP_NUM_THREADS='2',HF_HUB_OFFLINE='1')
    with (run/f'p2p_disabled_{setting}.log').open('w') as log:
        p=subprocess.Popen([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(root/'tmp/side_teacher_perf_20261029.py')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        begin=time.monotonic()
        try:
            while p.poll() is None:
                foreign=apps()-descendants(p.pid)
                if foreign or time.monotonic()-begin>600:raise RuntimeError(f'Conflict or timeout: {foreign}')
                time.sleep(3)
            if p.returncode:raise RuntimeError(f'Probe failed: {p.returncode}; inspect log')
        finally:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    time.sleep(5)
(run/'complete.json').write_text(json.dumps({'status':'complete'}))
