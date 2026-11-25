set -eu
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_joint4_uniform_1024
sed -i 's/\r$//' fetch.sh launch.sh
d=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
export PATH="$d/env/bin:$PATH" LD_LIBRARY_PATH="$d/runtime/lib:${LD_LIBRARY_PATH:-}"
nohup bash fetch.sh >fetch_retry.log 2>&1 </dev/null & echo FETCH_PID=$!
tail -5 run.log
