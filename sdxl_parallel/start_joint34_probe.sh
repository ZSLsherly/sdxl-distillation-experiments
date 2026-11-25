set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python -m py_compile sdxl_parallel/train.py
python -m unittest discover -s sdxl_parallel -p test_batching.py
run_stamp=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)
mkdir -p logs outputs/gan_stage_joint34
export CUDA_VISIBLE_DEVICES=0,1,2,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
nohup bash -c 'torchrun --standalone --nproc_per_node=4 sdxl_parallel/train.py --run "parallel_probe/joint34_4gpu_'"$run_stamp"'" --variant joint34 --data-dir datasets/parallel_10k_seed20261018 --batch 64 --micro-batch 16 --rollout-batch 16 --benchmark --benchmark-batch 64 --profile > "logs/joint34_probe_'"$run_stamp"'.log" 2>&1; echo $? > "logs/joint34_probe_'"$run_stamp"'.exit"' </dev/null >/dev/null 2>&1 &
probe_pid=$!
python - "$run_stamp" "$probe_pid" <<'PY'
import sys,json
from pathlib import Path
s,p=sys.argv[1:]
record={'timestamp':s,'launcher_pid':int(p),'run':f'parallel_probe/joint34_4gpu_{s}','log':f'logs/joint34_probe_{s}.log','exit':f'logs/joint34_probe_{s}.exit','world_size':4,'microbatch_per_gpu':16,'effective_batch':64,'student_optimizer_on_gpu':True}
Path('outputs/gan_stage_joint34/probe_launch.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record))
PY
