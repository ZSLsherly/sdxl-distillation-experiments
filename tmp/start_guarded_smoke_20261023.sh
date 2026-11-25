set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test ! -e outputs/gpu_smoke_20261023/default_b8_g4
mkdir -p outputs/gpu_smoke_20261023
nohup nice -n 10 ionice -c 2 -n 7 python run_guarded_smoke_20261023.py > outputs/gpu_smoke_20261023/guard.log 2>&1 </dev/null &
echo "$!" > outputs/gpu_smoke_20261023/guard.pid
echo "GUARD_STARTED $(cat outputs/gpu_smoke_20261023/guard.pid)"
