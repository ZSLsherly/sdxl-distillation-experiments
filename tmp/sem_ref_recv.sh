set -eu
d=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
cd "$d/outputs/semantic_references_1024/xfer"
nohup "$d/runtime/bin/python" -u xfer.py receive >transfer.log 2>&1 </dev/null & echo PID=$!
sleep 1;cat transfer.log
