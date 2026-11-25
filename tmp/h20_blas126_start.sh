set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_gan_mb32
/root/miniconda3/bin/python -m pip install --no-cache-dir --no-deps --target "$d/blas126" nvidia-cublas-cu12==12.6.4.1 --timeout 30 > "$d/blas_install.log" 2>&1
cd "$d/code"
L="$d/blas126/nvidia/cublas/lib"; export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12"
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python - <<'PY'
import torch
for n in (640,1280):
 for b in (8,32):
  for s,k in ((77,2048),(4096,n)):
   x=torch.zeros((b,s,k),device='cuda',dtype=torch.bfloat16);w=torch.zeros((n,k),device='cuda',dtype=x.dtype)
   y=torch.nn.functional.linear(x,w);torch.cuda.synchronize()
print('linear test OK; libraries=',sorted(set(l.split()[-1] for l in open('/proc/self/maps') if 'libcublas' in l)))
PY
sed -e 's/h20_gan_mb32_a4/h20_gan_mb32_blas126/g' -e "s/'gpu.csv'/'gpu_blas126.csv'/" run_h20.py > run_blas126.py
nohup python -u run_blas126.py </dev/null > blas126.log 2>&1 & echo $!
