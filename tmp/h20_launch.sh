set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_gan_mb32/code
mkdir -p "$d"
tar -xzf "$SDXL_WORK/tmp/h20_mb32.tar.gz" -C "$d"
cd "$d"
export PYTHONPATH="$d" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for i in $(seq 1 120); do test ! -s "$SDXL_WORK/flash_gan_inputs/sam-vit-base/model.safetensors" || break; sleep 3; done
test -s "$SDXL_WORK/flash_gan_inputs/sam-vit-base/model.safetensors"
python -m py_compile train.py run_h20.py
nohup python -u run_h20.py > h20.log 2>&1 </dev/null & echo $!
