set -eu
. /root/sdxl_research/env.sh
c=/root/autodl-tmp/h20_coco_after_g160
mkdir "$c"
cd "$c"
tar -xzf /root/autodl-tmp/h20_coco_jobs.tar.gz
export PYTHONPATH="$PWD:/root/autodl-tmp/h20_C_mix1_formal_20261103" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
L=/root/autodl-tmp/h20_gan_mb32/blas126/nvidia/cublas/lib
export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12"
cat > prepare_and_wait.sh <<'EOS'
set -eu
test "$(sha256sum /root/autodl-tmp/h20_coco10k_public_reference.tar | cut -d' ' -f1)" = 8f045bfbc2507ee74547665dadeb0eb87b33cfdc8805c83275089d926c5d2246
mkdir -p "$SDXL_WORK/bc_texture_inputs"
tar -xf /root/autodl-tmp/h20_coco10k_public_reference.tar -C "$SDXL_WORK/bc_texture_inputs"
python -c 'import cleanfid' || python -m pip install --no-deps --target "$PWD/vendor" clean-fid==0.1.35
export PYTHONPATH="$PWD/vendor:$PYTHONPATH"
python -c 'import evaluation_core'
python -u bc_public_sdxl_inputs.py
echo WAITING_FOR_G160
exec python -u after_g160.py
EOS
nohup bash prepare_and_wait.sh </dev/null > assessment.log 2>&1 & echo EVAL_MANAGER=$!
