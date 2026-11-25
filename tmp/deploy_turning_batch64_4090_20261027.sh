set -euo pipefail
ROOT=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
export LD_LIBRARY_PATH="$ROOT/runtime/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$ROOT/tmp"
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DEST="$ROOT/outputs/turning_batch64_g160_20261027/code"
test ! -e "$DEST"
mkdir -p "$DEST"
tar -xzf "$ROOT/tmp/turning_batch64_g160_20261027.tar.gz" -C "$DEST"
"$ROOT/env/bin/python" "$DEST/sdxl_research/test_turning_batch64_workflow.py"
"$ROOT/env/bin/python" "$DEST/sdxl_research/turning_batch64_workflow.py" --help >/dev/null
sha256sum "$DEST/sdxl_research/turning_batch64_workflow.py"
echo PREPARED_ONLY
