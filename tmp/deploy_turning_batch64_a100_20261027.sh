set -euo pipefail
ROOT=/data5/shenglin.zeng/sdxl_crossdistill
source /data5/shenglin.zeng/env.sh
export TMPDIR="$ROOT/tmp"
export CUDA_VISIBLE_DEVICES=''
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
DEST="$ROOT/outputs/turning_batch64_g160_20261027/code"
test ! -e "$DEST"
mkdir -p "$DEST"
tar -xzf "$ROOT/tmp/turning_batch64_g160_20261027.tar.gz" -C "$DEST"
/data5/shenglin.zeng/envs/crossdistill/bin/python "$DEST/sdxl_research/test_turning_batch64_workflow.py"
/data5/shenglin.zeng/envs/crossdistill/bin/python "$DEST/sdxl_research/turning_batch64_workflow.py" --help >/dev/null
sha256sum "$DEST/sdxl_research/turning_batch64_workflow.py"
echo PREPARED_ONLY
