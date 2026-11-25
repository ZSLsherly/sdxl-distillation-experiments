set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
out=outputs/turning_boundary_ablation_20261027
test ! -f "$out/paired_GPU_probe_complete.json"
if test -f "$out/shared_probe.pid" && kill -0 "$(cat "$out/shared_probe.pid")" 2>/dev/null;then echo 'Existing probe alive';exit 1;fi
nohup nice -n 10 python -u "$out/run_shared_probe.py" > "$out/shared_probe.log" 2>&1 < /dev/null &
echo $! > "$out/shared_probe.pid"
cat "$out/shared_probe.pid"
