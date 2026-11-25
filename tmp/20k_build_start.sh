set -eu
. /root/sdxl_research/env.sh
code="$SDXL_ROOT/tmp/prepare20k"
mkdir -p "$code"
tar -xzf "$SDXL_ROOT/tmp/prepare20k.tar.gz" -C "$code"
cd "$SDXL_WORK/datasets/20k_sources"
for i in $(seq 1 240); do
    if test -s vae_latents_002.pt && test -s vae_latents_003.pt; then break; fi
    if ! flock -n download.lock true; then sleep 10; else echo 'Downloads are inactive; see download.log'; exit 1; fi
done
test -s vae_latents_002.pt && test -s vae_latents_003.pt
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$code:$SDXL_ROOT/tmp/interval_D_20261103/code"
nohup flock -n build.lock nice -n 10 python -u "$code/build_20k_sdxl.py" </dev/null >> build.log 2>&1 &
echo CPU_BUILDER_PID=$!
