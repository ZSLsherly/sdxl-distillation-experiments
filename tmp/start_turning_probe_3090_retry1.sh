set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
out=outputs/turning_boundary_ablation_20261027
test -f "$out/data_verified.json"
test ! -d "$out/short_normalized_u_retry1"
nohup nice -n 10 env/bin/python -u "$out/run_shared_probe_retry1.py" > "$out/shared_probe_retry1.log" 2>&1 < /dev/null &
echo $! > "$out/shared_probe_retry1.pid"
cat "$out/shared_probe_retry1.pid"
