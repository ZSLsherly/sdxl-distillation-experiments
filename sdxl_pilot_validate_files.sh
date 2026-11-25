source /data5/shenglin.zeng/env.sh
python - <<'PY'
from pathlib import Path
import ast,json
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
for name in ['sdxl_hparam_train.py','sdxl_pilot_evaluate.py','sdxl_stage_metrics.py','sdxl_audit_pilot.py']:
 ast.parse((root/'src'/name).read_text())
print('SOURCES_PARSE_OK')
rows=[]
for rank in range(4):rows+=json.loads((root/f'outputs/hparam_pilot/lr5e7_pcm10/validation_after_rank{rank}.json').read_text())
print('VALIDATION_COUNTS',len(rows),len(set(x['cache_index'] for x in rows)),sorted(set(x['seed'] for x in rows)))
print('ALL_FINITE',all(x['finite'] for x in rows))
print('TRAINED_SHARDS',len(list((root/'outputs/hparam_pilot/lr5e7_pcm10').glob('student_rank*.pt'))))
PY
