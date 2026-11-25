set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import hashlib,json,shutil,subprocess
from pathlib import Path
p=Path('sdxl_turning_c_20261020')
ready=json.loads((p/'cpu_check_pass.json').read_text())
assert ready['archive_sha256']==hashlib.sha256(Path('turning_c_20261020.tar.gz').read_bytes()).hexdigest()
m=json.loads(Path('outputs/active_experiment.json').read_text())
for run in m['runs']:
    s=json.loads((Path('outputs')/run['run']/'status.json').read_text())
    assert s['status'] in ('complete','stopped'), ('existing experiment unfinished',run['run'],s)
free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
free=[int(x.strip()) for x in free.splitlines()]
assert len(free)>=4 and min(free[:4])>=36000, ('GPU memory changed; inspect before launching',free)
assert shutil.disk_usage('.').free > 200*1024**3, 'Need at least 200 GiB free for checkpoints'
print('PREFLIGHT_READY',free)
PY
run="turning_c_20261020/c_beta03_fresh_160"
mkdir -p outputs/turning_c_20261020
mkdir "outputs/$run"
mkdir -p logs
export CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
nohup torchrun --standalone --nproc_per_node=4 sdxl_turning_c_20261020/train.py \
 --run "$run" --variant joint4 --data-dir datasets/parallel_10k_seed20261018 \
 --batch 64 --micro-batch 16 --rollout-batch 16 \
 --max-student-updates 160 --save-every-updates 40 --snapshot-seconds 1800 \
 --eval-every-samples 128 --hours 24 \
 > logs/turning_c_beta03_20261020.log 2>&1 </dev/null &
train_pid=$!
python - "$run" "$train_pid" <<'PY'
import json,sys
from datetime import datetime,timezone,timedelta
from pathlib import Path
run,pid=sys.argv[1:]
m={
 'timestamp':datetime.now(timezone(timedelta(hours=8))).isoformat(),
 'experiment':'C_student_only_turning_mixture','beta':0.3,
 'world_size':4,'effective_batch':64,'max_student_updates':160,
 'initialization':'original SDXL Base; no resumed state',
 'baseline':'gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16',
 'runs':[{'variant':'joint4','run':run,'gpus':[0,1,2,3],
          'launcher_pid':int(pid),'log':'logs/turning_c_beta03_20261020.log'}]
}
Path('outputs/turning_c_20261020/launch.json').write_text(json.dumps(m,indent=2))
print(json.dumps(m))
PY
