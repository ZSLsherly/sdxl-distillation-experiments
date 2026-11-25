R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
"$R/env/bin/python" - <<'PY'
import json,hashlib,inspect
from pathlib import Path
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
r=Path('/data1/shenglin.zeng/dmd2_matched_10k_20261119')
names=['code/train_matched.py','code/vendor/main/sd_guidance.py','code/vendor/main/sd_unified_model.py','code/vendor/main/sd_unet_forward.py','code/vendor/main/utils.py']
print(json.dumps({'remote_sha256':{x:hashlib.sha256((r/x).read_bytes()).hexdigest() for x in names},'run_config_sha256':hashlib.sha256((r/'runs/formal_g160_restart_20261120/config.json').read_bytes()).hexdigest(),'run_log_sha256':hashlib.sha256((r/'runs/formal_g160_restart_20261120/rank0.jsonl').read_bytes()).hexdigest()},indent=2))
print('MIXED_PRECISION_POLICY_SOURCE')
print(inspect.getsource(MixedPrecisionPolicy))
PY
