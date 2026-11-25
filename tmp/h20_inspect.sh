set -eu
date
nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu,driver_version --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
df -h / /root/autodl-tmp; ls -la /root/autodl-tmp | head -32
ls -d /root/miniconda* /root/sdxl* /root/autodl-fs* || true
cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true
for p in /root/miniconda3/bin/python /usr/bin/python3; do if test -x "$p"; then "$p" -c "import sys,importlib.metadata as m;print(sys.executable,sys.version);print([(k,m.version(k)) for k in ['torch','torchvision','diffusers','transformers','accelerate','safetensors','numpy'] if any(d.metadata['Name'].lower()==k for d in m.distributions())])"; fi; done
