set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test ! -e outputs/joint4_uniform_10k_2gpu_mb16_20261023
mkdir -p outputs/joint4_uniform_10k_2gpu_mb16_20261023
nohup nice -n 10 ionice -c 2 -n 7 python resume_full10k_mb16_20261023.py > outputs/joint4_uniform_10k_2gpu_mb16_20261023/guard.log 2>&1 </dev/null &
echo "$!" > outputs/joint4_uniform_10k_2gpu_mb16_20261023/guard.pid
echo "FULL_TRAINING_SUPERVISOR_STARTED $(cat outputs/joint4_uniform_10k_2gpu_mb16_20261023/guard.pid)"

