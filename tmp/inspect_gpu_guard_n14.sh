ps -p 3357 -o pid,user,etime,args
/data1/shenglin.zeng/sdxl_eval_uniform_20261025/env/bin/python - <<'PY'
import json,pynvml as n
n.nvmlInit()
for u in json.load(open('/data1/shenglin.zeng/high30_parallel_eval_20261114/node.json'))['gpus']:
 h=n.nvmlDeviceGetHandleByUUID(u)
 print(u,'COMPUTE',[(p.pid,p.usedGpuMemory) for p in n.nvmlDeviceGetComputeRunningProcesses(h)],'GRAPHICS',[(p.pid,p.usedGpuMemory) for p in n.nvmlDeviceGetGraphicsRunningProcesses(h)])
PY
