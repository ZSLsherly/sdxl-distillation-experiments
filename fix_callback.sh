set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT/src/SparkDiffusion"
python - <<'PY'
from pathlib import Path
p=Path('sparkdiffusion/callbacks/every_n_draw_distill.py');s=p.read_text();s=s.replace('from megatron.core import parallel_state','try:\n    from megatron.core import parallel_state\nexcept ImportError:\n    parallel_state = None');p.write_text(s)
PY
python -c 'from sparkdiffusion.configs.registry_distill import make_config; c=make_config(); print("CONFIG_IMPORT_OK")'
tail -8 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
