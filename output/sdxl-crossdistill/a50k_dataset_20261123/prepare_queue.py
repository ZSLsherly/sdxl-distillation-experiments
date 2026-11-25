"""Wait for verified transfers, then build data on one idle GPU; never train."""
import fcntl, json, subprocess, time
from pathlib import Path
root=Path('/data/shenglin.zeng/a50k_20261123')
lock=(root/'records/prepare_queue.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def status(state,**extra):
    value=dict(state=state,unix_time=time.time(),**extra)
    (root/'records/prepare_status.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)
deadline=time.monotonic()+6*3600
status('waiting_for_download')
while time.monotonic()<deadline:
    source=root/'datasets/sources/sources.json'
    if source.exists():
        specs=[s for s in json.loads(source.read_text())['files'] if not s['fallback_only']]
        if all((root/'datasets/sources'/s['name']).is_file()
               and (root/'datasets/sources'/s['name']).stat().st_size==s['size'] for s in specs):
            break
    time.sleep(30)
else:status('download_wait_timeout');raise SystemExit(1)
status('waiting_for_idle_gpu')
while time.monotonic()<deadline:
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    busy=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    busy={line.split(',')[0].strip() for line in busy.splitlines()}
    free=[]
    for row in rows.splitlines():
        index,uuid,used,util=[s.strip() for s in row.split(',')]
        if index in ('1','2','3','4','5') and uuid not in busy and int(used)<256 and int(util)<=5:free.append(index)
    if free:break
    time.sleep(30)
else:status('gpu_wait_timeout');raise SystemExit(1)
status('building',physical_gpu=free[0])
rc=subprocess.run(['bash',str(root/'code/build_start.sh'),free[0]]).returncode
validation=root/'datasets/parallel_50k_seed20261018/validation.json'
if rc or not validation.exists():status('failed',exit_code=rc,log=str(root/'records/build.log'));raise SystemExit(1)
v=json.loads(validation.read_text())
assert v['train']==50000 and v['holdout']==32 and v['old_samples_bit_identical']
status('ready',dataset=str(validation.parent),validation=v)
