set -eu
. /root/sdxl_research/env.sh
b="$SDXL_ROOT/outputs/bc_winner_late_mix_20261102"
ls -ld "$SDXL_WORK/bc_texture_inputs" "$SDXL_ROOT"/tmp/*coco* "$b/coco10k_C_mix1"
ls "$b/coco10k_C_mix1"
cat "$b/coco10k_C_mix1/complete.json" | head -c 1200
ls "$SDXL_WORK/bc_texture_inputs"
ls -l "$SDXL_ROOT/tmp/bc_coco_winner_20261102/code/" | grep -E 'evaluate|evaluation_core|control_after|coco_then'
ls "$SDXL_WORK/bc_texture_inputs/reference_features/real" | wc -l
du -sh "$SDXL_WORK/bc_texture_inputs"
python - <<'PY'
from pathlib import Path
import json,os
b=Path(os.environ['SDXL_ROOT'])/'outputs/bc_winner_late_mix_20261102'
m=json.loads((b/'coco10k_C_mix1/manifest.json').read_text()); print(m.keys(),list(m['models']))
print(json.loads((b/'late_mix/status.json').read_text()).keys())
PY

