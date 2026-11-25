set -eu
. /root/sdxl_research/env.sh
df -h "$SDXL_ROOT" "$SDXL_WORK"
find "$SDXL_ROOT/models" "$SDXL_WORK" -maxdepth 4 \( -iname '*sam-vit*' -o -iname '*sam_vit*' \) -print | head -36
ls "$SDXL_ROOT/models"
python - <<'PY'
import importlib.metadata as m
print('transformers',m.version('transformers'))
PY
