set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT/src/SparkDiffusion"
python - <<'PY'
from pathlib import Path
p=Path('sparkdiffusion/models/t2v_model_finetune.py');s=p.read_text();s=s.replace('from sparkdiffusion.utils.fused_adam_dtensor import FusedAdam','try:\n    from sparkdiffusion.utils.fused_adam_dtensor import FusedAdam\nexcept ImportError:\n    FusedAdam = None');p.write_text(s)
PY
python -c 'from sparkdiffusion.configs.registry_distill import make_config; c=make_config(); print("CONFIG_IMPORT_OK")'
tail -8 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
