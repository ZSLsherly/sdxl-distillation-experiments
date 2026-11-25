"""One G160 experiment using the completed C+1% recipe with both middle DMs."""
import fcntl,json,math,os,shutil,subprocess,time
from pathlib import Path
from launch import command,gpu_check
from run_quota_bc import verify
from turning_batch64_workflow import descendants,foreign_on,stop_own
root=Path(os.environ['SDXL_ROOT']);out=root/'outputs/c_mix1_dm234_20261103';code=Path(__file__).resolve().parent
def main():
    lock=(out/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=json.loads((out/'effective_config.json').read_text())
    path=out/'model'
    if path.exists():raise RuntimeError('Existing run present')
    print('QUEUED',flush=True)
    while True:
        if (out/'STOP_REQUEST').exists(): return
        if shutil.disk_usage(out).free>=7*1024**3:
            try:cards=gpu_check([0],76);break
            except RuntimeError:pass
        time.sleep(30)
    with (out/'train.log').open('x') as f:
        p=subprocess.Popen(command(cfg,root,out.name+'/model',1),cwd=code,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        print('STARTED',p.pid,flush=True)
        try:
            while p.poll() is None:
                if foreign_on({cards[0][1]},descendants(p.pid)):raise RuntimeError('Foreign GPU process')
                time.sleep(10)
            if p.returncode:raise RuntimeError(f'Training failed ({p.returncode})')
        except BaseException:stop_own(p);raise
    w=verify(path)
    logs=[json.loads(l) for l in (path/'rank0.jsonl').read_text().splitlines()]
    gs=[r for r in logs if r['phase']=='student_done']; fs=[r for r in logs if r['phase']=='fake']
    assert all(r['supervised_stages']==[2,3,4] and not r['gan'] for r in gs)
    assert [r['fake_supervised_stages'] for r in fs]==[[2,3],[4]]*160
    assert json.loads((Path(w)/'cpu_validation.json').read_text())['strict_load']
    (out/'complete.json').write_text(json.dumps(dict(G=160,fake=320,checkpoint=w)))
if __name__=='__main__':main()
