R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
CUDA_VISIBLE_DEVICES=0 "$R/env/bin/python" - <<'PY'
import ctypes
c=ctypes.CDLL('libcuda.so.1');v=c.cuInit(0);msg=ctypes.c_char_p();c.cuGetErrorString(v,ctypes.byref(msg));print('CUDA_DRIVER_INIT',v,msg.value)
if v==0:
 import torch
 x=torch.ones(1,device='cuda');print('CUDA_TENSOR_OK',x.item())
PY
ps -eo pid,args | grep -E '[g]pu_probe.py|[r]un_guard.py|[t]rain_matched.py' || true
