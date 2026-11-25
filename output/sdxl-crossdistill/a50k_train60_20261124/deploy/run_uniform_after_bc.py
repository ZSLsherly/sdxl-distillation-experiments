"""Wait for the active B/C queue, then run its matched uniform-time control."""
import fcntl,json,os,shutil,subprocess,time
from run_quota_bc import ROOT,CODE,OUT,verify
from launch import command,gpu_check
from turning_batch64_workflow import descendants,foreign_on,stop_own

def main():
    own=(OUT/'uniform_queue.lock').open('a')
    fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    prior=(OUT/'queue.lock').open('a')
    print('WAITING_FOR_BC',flush=True)
    while True:
        if any((OUT/s).exists() for s in ['USER_STOP','STOP_REQUEST']):
            raise RuntimeError('Stop requested; not starting control')
        try:
            fcntl.flock(prior,fcntl.LOCK_EX|fcntl.LOCK_NB)
            break
        except BlockingIOError:
            time.sleep(30)
    if not (OUT/'queue_complete.json').is_file():
        raise RuntimeError('B/C queue ended without completion; inspect before proceeding')
    for arm in ['B','C']:
        verify(OUT/arm)
    path=OUT/'uniform_control'
    if path.exists():
        print('ALREADY_COMPLETE',verify(path),flush=True)
        return
    config=json.loads((CODE/'configs/uniform_control_a800_g160.json').read_text())
    reference=json.loads((CODE/'configs/turning_C_quota_a800_g160.json').read_text())
    expected=dict(reference,turning_profile=None,match_fake_prior=False)
    assert config==expected and config['pair_fake_stages']
    if shutil.disk_usage(OUT).free<12*1024**3:
        raise RuntimeError('Need 12GiB free before control')
    for attempt in range(12):
        try:
            cards=gpu_check([0],76)
            break
        except RuntimeError:
            if attempt==11: raise
            time.sleep(5)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with (OUT/'uniform_control.log').open('x') as log:
        proc=subprocess.Popen(command(config,ROOT,f'{OUT.name}/uniform_control',1),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'event':'started','arm':'uniform_control','pid':proc.pid,'gpus':cards}),flush=True)
        try:
            while proc.poll() is None:
                if foreign_on({r[1] for r in cards},descendants(proc.pid)) or (OUT/'USER_STOP').exists():
                    raise RuntimeError('Foreign GPU process or user stop')
                time.sleep(10)
            if proc.returncode:
                raise RuntimeError(f'Control failed: exit {proc.returncode}')
        except BaseException:
            stop_own(proc)
            raise
    final=verify(path)
    (OUT/'uniform_control_complete.json').write_text(json.dumps({'final_weights':final},indent=2))
    print('UNIFORM_CONTROL_COMPLETE',final,flush=True)

if __name__=='__main__':main()
