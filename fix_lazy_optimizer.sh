set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT/src/SparkDiffusion"
python - <<'PY'
from pathlib import Path
p=Path('sparkdiffusion/utils/optim_instantiate_dtensor.py')
s=p.read_text();s=s.replace('from sparkdiffusion.utils.fused_adam_dtensor import FusedAdam\n','')
s=s.replace('elif optim_type == "fusedadam":\n        opt_cls = FusedAdam','elif optim_type == "fusedadam":\n        from sparkdiffusion.utils.fused_adam_dtensor import FusedAdam\n        opt_cls = FusedAdam')
p.write_text(s)
PY
python -c 'from sparkdiffusion.configs.registry_distill import make_config; c=make_config(); print("CONFIG_IMPORT_OK")'
tail -5 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
