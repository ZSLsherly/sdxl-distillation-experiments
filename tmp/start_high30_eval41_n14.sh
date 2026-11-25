set -eu
r=/data1/shenglin.zeng/high30_parallel_eval_20261114
p=/data1/shenglin.zeng/sdxl_eval_uniform_20261025/env/bin/python
cd "$r"
"$p" - <<'PY'
import sys,json
from pathlib import Path
sys.path.insert(0,'code')
from resource_guard import status
root=Path('.')
assert (root/'assets_ready.json').exists()
assert not (root/'STOP_REQUEST').exists()
for u in json.loads((root/'node.json').read_text())['gpus']:
 pids,free,util=status(u)
 print(u,'pids',sorted(pids),'free_GiB',round(free/2**30,2),'util',util)
 assert not pids and free>20*2**30 and util<=5,'Selected GPU is not idle'
PY
nohup env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 "$p" -u code/run_parallel.py "$r" >queue.log 2>&1 </dev/null &
echo $! >queue.pid
cat queue.pid
