set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_gan_mb32/code
cd "$d"
python - <<'PY'
import torch
print(torch.__version__,torch.backends.cudnn.version(),torch.cuda.get_device_properties(0))
PY
export PYTHONFAULTHANDLER=1 TORCH_SHOW_CPP_STACKTRACES=1 CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
sed -e 's/h20_gan_mb32_a4/h20_gan_mb32_diag/g' -e "s/'gpu.csv'/'gpu_diag.csv'/" run_h20.py > run_diag.py
nohup python -u run_diag.py </dev/null > diag.log 2>&1 & echo $!
