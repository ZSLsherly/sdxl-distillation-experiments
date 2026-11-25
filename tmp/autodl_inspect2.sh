printf '%s\n' "$PATH"
ls -d /root/miniconda* /opt/conda* /root/autodl* 2>/dev/null || true
ls /root/miniconda3/bin/python* 2>/dev/null || true
/root/miniconda3/bin/python - <<'PY'
import sys,importlib.metadata as m
print(sys.version)
for p in ['torch','torchvision','diffusers','transformers','accelerate','numpy','scipy','safetensors','Pillow']:
 try: print(p,m.version(p))
 except m.PackageNotFoundError: print(p,'missing')
import torch
print('CUDA_BUILD',torch.version.cuda,'GPU_AVAILABLE',torch.cuda.is_available())
PY
