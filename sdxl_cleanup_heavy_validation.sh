set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import shutil,json
root=Path('/data5/shenglin.zeng/sdxl_crossdistill').resolve()
# Only the large temporary checkpoint made by this now-cancelled code test.
p=(root/'outputs/v4_validation/control_20261017T205945/checkpoints').resolve()
assert p.parent==(root/'outputs/v4_validation/control_20261017T205945').resolve() and p.name=='checkpoints'
size=sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) if p.exists() else 0
if p.exists():shutil.rmtree(p)
removed=[]
for name in ['sdxl_training_state.py','sdxl_state_test.py','sdxl_compare_resumed_state.py','sdxl_validation_minimal_supervisor.py','sdxl_run_v4_validation.sh']:
 f=(root/'src'/name).resolve();assert f.parent==(root/'src').resolve()
 if f.exists():f.unlink();removed.append(name)
record={'removed_temporary_checkpoint_bytes':size,'removed_unused_runtime_files':removed,'kept':'logs and diagnostic images; base models, datasets and formal experiment outputs untouched'}
(root/'outputs/v4_validation/cleanup.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
PY
