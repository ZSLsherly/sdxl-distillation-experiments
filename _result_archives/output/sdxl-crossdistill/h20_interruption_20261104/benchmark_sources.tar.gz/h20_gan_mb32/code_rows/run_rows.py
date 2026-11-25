import json,subprocess,sys,os,fcntl,time
from pathlib import Path
from launch import command,gpu_check
from turning_batch64_workflow import descendants,foreign_on,stop_own
d=Path(__file__).resolve().parent; root=Path(os.environ['SDXL_ROOT']); out=root/'outputs/h20_gan_mb32_rows'
lock=(d/'job.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
cfg=json.loads((d/'config.json').read_text())
assert cfg['micro_batch']==32 and cfg['batch']==cfg['benchmark_batch']==128 and cfg['benchmark_cycles']==3
cards=gpu_check([0],90)
assert not out.exists()
gpu_log=d/'gpu.csv'
with gpu_log.open('x') as f:
    sampler=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,memory.used,utilization.gpu','--format=csv,noheader,nounits','-l','1'],stdout=f)
    try:
        p=subprocess.Popen(command(cfg,root,out.name,1),start_new_session=True)
        try:
            while p.poll() is None:
                if foreign_on({cards[0][1]},descendants(p.pid)):raise RuntimeError('GPU conflict')
                time.sleep(3)
            if p.returncode:sys.exit(p.returncode)
        except (Exception,KeyboardInterrupt):stop_own(p);raise
    finally:sampler.terminate();sampler.wait()
r=json.loads((out/'benchmark.json').read_text());assert (r['student_updates'],r['fake_updates'],r['D_updates'],r['microbatch_per_gpu'],r['accumulation'])==(3,6,3,32,4)
print('COMPLETE',r,flush=True)
