set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
m=json.loads(Path('outputs/gan_stage_joint34/probe_launch.json').read_text())
assert Path(m['exit']).read_text().strip()=='0', 'Probe must pass before training'
b=json.loads((Path('outputs')/m['run']/'benchmark.json').read_text())
assert b['student_updates']==3 and b['effective_batch']==64
rows=[json.loads(s) for s in (Path('outputs')/m['run']/'rank0.jsonl').read_text().splitlines()]
stages=[(r['stage'],r['dmd'],r['gan']) for r in rows if r['phase']=='student_done']
assert stages==[(2,True,False),(3,True,True),(4,True,True)],stages
print('PROBE_PASSED',stages,'peak_GiB',b['peak_allocated_GiB'])
PY
run_stamp=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)
export CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
nohup torchrun --standalone --nproc_per_node=4 sdxl_parallel/train.py \
 --run "gan_stage_joint34/joint34_${run_stamp}_4gpu_mb16" --variant joint34 \
 --data-dir datasets/parallel_10k_seed20261018 --batch 64 --micro-batch 16 --rollout-batch 16 \
 --max-student-updates 160 --save-every-updates 0 --snapshot-seconds 1800 --eval-every-samples 128 --hours 24 \
 > "logs/joint34_${run_stamp}.log" 2>&1 </dev/null &
train_pid=$!
python - "$run_stamp" "$train_pid" <<'PY'
import sys,json
from pathlib import Path
s,p=sys.argv[1:]
m={'timestamp':s,'world_size':4,'microbatch_per_gpu':16,'rollout_batch':16,'effective_batch':64,'accumulation':1,
   'max_student_updates':160,'save_every_updates':0,'snapshot_seconds':1800,'eval_every_updates':2,'student_optimizer_on_gpu':True,
   'data_dir':'datasets/parallel_10k_seed20261018','fresh_initialization':'original SDXL Base',
   'gan_stages':[3,4],'student_lr':1e-6,'fake_lr':1e-6,'gan_weight':0.005274229682981968,
   'runs':[{'variant':'joint34','run':f'gan_stage_joint34/joint34_{s}_4gpu_mb16','gpus':[0,1,2,3],
            'launcher_pid':int(p),'log':f'logs/joint34_{s}.log'}]}
for path in ['outputs/gan_stage_joint34/launch.json','outputs/active_experiment.json']:
 Path(path).write_text(json.dumps(m,indent=2))
print(json.dumps(m))
PY
