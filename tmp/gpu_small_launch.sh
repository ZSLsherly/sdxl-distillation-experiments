set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import sys
sys.path.insert(0,'sdxl_research')
from launch import gpu_check
print('IDLE_CHECK',gpu_check([0,3],36))
PY
mkdir -p outputs/research_small_20261022 logs
export CUDA_VISIBLE_DEVICES=0,3 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=2
nohup nice -n 10 ionice -c 2 -n 7 bash -c '
torchrun --standalone --nproc_per_node=2 sdxl_research/test_training_state_gpu.py outputs/research_small_20261022/toy_state
rc=$?
echo "$rc" > outputs/research_small_20261022/toy_state.exit
exit "$rc"
' > logs/research_small_state_20261022.log 2>&1 </dev/null &
echo "$!" > outputs/research_small_20261022/toy_launcher.pid
echo "STARTED_TOY_STATE_CHECK $(cat outputs/research_small_20261022/toy_launcher.pid)"
