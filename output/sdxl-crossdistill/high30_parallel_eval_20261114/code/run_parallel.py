"""Independent rank workers; preserve completed blocks and yield to foreign GPU tasks."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fcntl,json,os,signal,subprocess,sys,time
from resource_guard import status
ROOT=Path(sys.argv[1]);CODE=Path(__file__).resolve().parent
node=json.loads((ROOT/'node.json').read_text());gpus=node['gpus'];arm='A_clean_global_high30_v2';out=ROOT/'results'/arm
def stop(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=15)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
def rank_worker(rank,uuid):
    marker=out/f'worker_{rank}_done.json'
    while not marker.exists():
        if (ROOT/'STOP_REQUEST').exists():raise RuntimeError('Evaluation stop requested')
        pids,free,util=status(uuid)
        if pids or free<20*2**30 or util>5:
            time.sleep(20);continue
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        with (out/f'worker_{rank}.log').open('a') as log:
            p=subprocess.Popen([sys.executable,'-u',str(CODE/'worker.py'),str(ROOT),str(rank),str(len(gpus))],
                env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('START',rank,uuid,p.pid,flush=True);conflict=False
            try:
                while p.poll() is None:
                    if (ROOT/'STOP_REQUEST').exists():stop(p);raise RuntimeError('Evaluation stop requested')
                    if status(uuid)[0]-{p.pid}:
                        stop(p);conflict=True;print('YIELD_GPU',rank,flush=True);break
                    time.sleep(5)
                rc=p.wait()
            except BaseException:
                (ROOT/'STOP_REQUEST').touch();stop(p);raise
        if conflict or rc==75:time.sleep(20);continue
        if rc:
            (ROOT/'STOP_REQUEST').touch()
            raise RuntimeError(f'rank {rank} failed: {rc}; cached feature blocks retained')
    print('RANK_COMPLETE',rank,flush=True)
if __name__=='__main__':
    with (ROOT/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert (ROOT/'assets_ready.json').exists()
        if not (out/'all_complete.json').exists():
            with ThreadPoolExecutor(len(gpus)) as pool:list(pool.map(lambda x:rank_worker(*x),enumerate(gpus)))
            with (out/'metrics.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(CODE/'worker.py'),str(ROOT),'finalize'],
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2'),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
        print('EVALUATION_COMPLETE',flush=True)
