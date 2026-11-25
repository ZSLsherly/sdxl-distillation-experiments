"""Two identical 24-G C pilots, run only after the matching uniform control."""
import fcntl,json,os,subprocess,time,math
from pathlib import Path
from launch import command,gpu_check
from turning_batch64_workflow import descendants,foreign_on,stop_own
from run_quota_bc import verify

root=Path(os.environ['SDXL_ROOT'])
previous=root/'outputs/turning_BC_paired_quota_20261102'
out=root/'outputs/fake_tracking_pilot_20261102'


def completed(path):
    progress=json.loads((path/'status.json').read_text())
    assert (progress['status'],progress['student_updates'],progress['fake_updates'])==('complete',24,48)
    rows=[json.loads(l) for l in (path/'rank0.jsonl').read_text().splitlines()]
    checks=[r for r in rows if r['phase']=='heldout_fake_tracking']
    assert [r['g'] for r in checks]==[8,16,24]
    for check in checks:
        assert [r['time'] for r in check['results']]==[60,499,940]
        assert all(math.isfinite(v) for r in check['results'] for v in r.values())
        e=next((path/'eval').glob(f'*_g{check["g"]:06d}_pilot/rank0.json'))
        cases=json.loads(e.read_text())['samples']
        assert len(cases)==8 and all((e.parent/c['image']).is_file() for c in cases)
    return checks


def main():
    out.mkdir(exist_ok=True)
    lock=(out/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    gate=(previous/'uniform_queue.lock').open('a')
    print('WAITING_FOR_UNIFORM_G160',flush=True)
    fcntl.flock(gate,fcntl.LOCK_EX)
    assert (previous/'uniform_control_complete.json').is_file()
    for arm in ('B','C','uniform_control'): verify(previous/arm)
    base=json.loads((root/'code/sdxl_research/configs/turning_C_quota_a800_g160.json').read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',TORCHDYNAMO_DISABLE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    results={}
    for name,rate in [('control',0.),('mix_1pct',.01)]:
        path=out/name
        if path.exists():
            results[name]=completed(path)
            continue
        if any((d/s).exists() for d in (out,previous) for s in ('STOP_REQUEST','USER_STOP')):
            raise RuntimeError('Stop requested')
        cards=gpu_check([0],76)
        cfg=dict(base,benchmark=True,benchmark_cycles=24,benchmark_batch=64,benchmark_warmup=1,
                 max_student_updates=24,fake_track_pilot=True,fake_student_mix=rate)
        with (out/f'{name}.log').open('x') as log:
            p=subprocess.Popen(command(cfg,root,f'{out.name}/{name}',1),cwd=root,env=env,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('STARTED',name,p.pid,flush=True)
            try:
                while p.poll() is None:
                    if foreign_on({r[1] for r in cards},descendants(p.pid)) or (out/'USER_STOP').exists():
                        raise RuntimeError('GPU conflict or immediate stop')
                    time.sleep(10)
                if p.returncode:raise RuntimeError(f'{name} exited {p.returncode}')
            except BaseException:
                stop_own(p)
                raise
        results[name]=completed(path)
    (out/'probe_results.json').write_text(json.dumps(results,indent=2))
    print('PILOT_COMPLETE',flush=True)


if __name__=='__main__':main()
