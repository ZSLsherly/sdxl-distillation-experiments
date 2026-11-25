set -eu
base=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
ls -d "$base"/models/* "$base"/outputs/coco10k/features/* /data1/shenglin.zeng/clip_missing_20261113/* 2>/dev/null
CUDA_VISIBLE_DEVICES=GPU-dee80cca-d29a-9086-d006-b76ba6ee093e "$base/env/bin/python" - <<'PY'
import json,torch
try:
 import pynvml as n
 n.nvmlInit()
 uuids=['GPU-dee80cca-d29a-9086-d006-b76ba6ee093e','GPU-7328ece9-841e-484d-9e72-d5b762fdce95','GPU-c374184a-ab5a-681a-ed48-025bed2da39b','GPU-4fba4d7f-66e5-aa77-7210-3803fe7190df','GPU-baca962f-0eff-b198-1677-8c28d525ecc1','GPU-df453630-a9d5-be2e-43db-e0abc1834da2']
 for u in uuids:
  h=n.nvmlDeviceGetHandleByUUID(u)
  print('NVML',u,'free',n.nvmlDeviceGetMemoryInfo(h).free,'pids',[p.pid for p in n.nvmlDeviceGetComputeRunningProcesses(h)])
 h=n.nvmlDeviceGetHandleByUUID(uuids[0])
 assert not n.nvmlDeviceGetComputeRunningProcesses(h),'GPU1 became busy'
except ImportError as e:
 print('MISSING_NVML',e)
 raise SystemExit(0)
print('TORCH',torch.__version__,'devices',torch.cuda.device_count())
x=torch.ones(16,device='cuda');print('CUDA_TEST',x.sum().item(),torch.cuda.get_device_name())
PY
