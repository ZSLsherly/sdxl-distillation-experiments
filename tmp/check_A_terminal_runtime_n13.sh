set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import json,hashlib
r=Path('/root/sdxl_research/outputs/A_pcm10_terminal_low30_g160_20261113');base=Path('/root/sdxl_research/outputs/A_pcm10_g160/model');code=Path('/root/sdxl_research/tmp/A_pcm10_terminal_low30_g160_20261113/code')
c=json.loads((r/'model/config.json').read_text());a=json.loads((base/'config.json').read_text())
keys=['schedule','pcm_weight','fake_weight_max','student_lr','fake_lr','fake_updates','batch','micro_batch','rollout_batch','teacher_cfg','seed','grad_clip','checkpoint_mode','optimizer_settings']
assert all(c[k]==a[k] for k in keys);assert c['turning_profile'] and c['match_fake_prior']
x=json.loads((r/'model/data_manifest.json').read_text());y=json.loads((base/'data_manifest.json').read_text());assert x['train_original_indices']==y['train_original_indices']
print('RUNTIME_CONFIG_AND_DATA_MATCH_A')
s=(code/'train.py').read_text();print('LOG_FIELDS',{k:k in s for k in ['supervised_stages','fake_grad_norm','boundary_vs_final_rms','terminal_dm_loss','middle_dm_loss']})
print('SOURCE_DIFF',[p.name for p in (base/'source').glob('*.py') if p.read_bytes()!=(code/p.name).read_bytes()])
PY
