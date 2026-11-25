set -eu
cd /root/sdxl_research
ls -la
cat env.sh
du -sh datasets/* models/* 2>/dev/null || true
ls -l models/sdxl-base-1.0/unet | head
. ./env.sh
python -c 'import torch,transformers,diffusers;print(torch.__version__,transformers.__version__,diffusers.__version__,torch.cuda.get_device_name())'
