set -eu
uname -a
df -h / /root/autodl-tmp /root/autodl-fs 2>/dev/null || true
free -h
command -v python
python - <<'PY'
import sys,importlib.metadata as m
print(sys.version)
for p in ['torch','torchvision','diffusers','transformers','accelerate','numpy','scipy','safetensors','Pillow']:
 try: print(p,m.version(p))
 except m.PackageNotFoundError:print(p,'missing')
import torch
print('CUDA_BUILD',torch.version.cuda,'GPU_AVAILABLE',torch.cuda.is_available())
PY
ls -la /root/autodl-tmp /root/autodl-fs 2>/dev/null || true
