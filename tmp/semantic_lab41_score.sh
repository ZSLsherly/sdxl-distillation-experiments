set -eu
d=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
cd "$d/outputs/semantic_joint4_uniform_1024"
mkdir -p model
tar -xf /data1/shenglin.zeng/public_clip_B32_20261104.tar -C model
test "$(sha256sum model/pytorch_model.bin | cut -d' ' -f1)" = a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f
export PATH="$d/env/bin:$PATH" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 LD_LIBRARY_PATH="$d/runtime/lib:${LD_LIBRARY_PATH:-}" CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup bash -c 'set -eu; until test "$(find images -name "*.png" | wc -l)" -eq 896; do sleep 5; done; exec python -u score.py' >score.log 2>&1 </dev/null & echo CPU_SCORE_PID=$!
