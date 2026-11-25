set -eu
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
import json
from pathlib import Path
p=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/turning_boundary_ablation_20261027')
log=p/'short_normalized_u/rank0.jsonl'
rows=[json.loads(s) for s in log.read_text().splitlines() if s.strip()] if log.exists() else []
done=[r for r in rows if r.get('phase')=='student_done'];fake=[r for r in rows if r.get('phase')=='fake_done' or r.get('phase')=='fake']
times=[r for r in rows if r.get('phase')=='boundary_turning_sample']
result={'status':'stopped_host_RAM_reserve','student_updates_completed':len(done),'fake_update_records':len(fake),
 'sampled_boundary_records':times,'error':json.loads((p/'paired_GPU_probe_error.json').read_text()),
 'raw_angle_started':(p/'short_raw_angle').exists(),'formal_training_started':False}
(p/'probe_summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
PY
