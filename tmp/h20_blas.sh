set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_gan_mb32/code
sed -n '2742,2752p' "$SDXL_ROOT/env/lib/python3.12/site-packages/diffusers/models/attention_processor.py"
ls -d /usr/local/cuda* /root/miniconda3/lib/python*/site-packages/nvidia/cublas/lib || true
python - <<'PY'
import torch
a=torch.randn(8,4096,640,device='cuda',dtype=torch.bfloat16); print('libs',sorted(set(l.split()[-1] for l in open('/proc/self/maps') if 'cublas' in l)))
PY
