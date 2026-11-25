set -eu
R=/data/shenglin.zeng/a50k_20261123
PY=/data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python
mkdir -p "$R/records"
export CUDA_VISIBLE_DEVICES="$1" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -m py_compile "$R/code/build_50k.py" "$R/code/sdxl_prepare.py"
cd "$R/code"
flock -n "$R/records/build.lock" "$PY" -u build_50k.py \
 --old-dir /data3/shenglin.zeng/sdxl_turning_ablation_20261028/datasets/parallel_10k_seed20261018 \
 --source-dir "$R/datasets/sources" \
 --model /data3/shenglin.zeng/clip_missing_20261113/models/sdxl-base-1.0 \
 --coco-manifest "$R/datasets/coco10k_reference.json" \
 --output "$R/datasets/parallel_50k_seed20261018" \
 --device cuda > "$R/records/build.log" 2>&1
tail -6 "$R/records/build.log"
