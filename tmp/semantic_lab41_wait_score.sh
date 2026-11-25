set -eu
d=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
cd "$d/outputs/semantic_joint4_uniform_1024"
cat > score_when_ready.sh <<'SH'
set -eu
until test "$(stat -c %s /data1/shenglin.zeng/public_clip_B32_20261104.tar)" = 606648320; do sleep 5; done
mkdir -p model
tar -xf /data1/shenglin.zeng/public_clip_B32_20261104.tar -C model
test "$(sha256sum model/pytorch_model.bin | cut -d' ' -f1)" = a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f
until test "$(find images -name '*.png' | wc -l)" = 896; do sleep 5; done
echo SCORING_ALL_1024
exec python -u score.py
SH
export PATH="$d/env/bin:$PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LD_LIBRARY_PATH="$d/runtime/lib:${LD_LIBRARY_PATH:-}" CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup bash score_when_ready.sh </dev/null >score.log 2>&1 & echo SCORE_PID=$!
