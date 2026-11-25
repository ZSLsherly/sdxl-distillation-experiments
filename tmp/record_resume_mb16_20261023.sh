set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import json,math,time
root=Path('outputs/joint4_uniform_10k_2gpu_mb16_20261023');p=root/'train'
c=json.loads((p/'config.json').read_text())
assert c['resume_repack'] and c['resume'] and not c['benchmark']
assert c['micro_batch']==16 and c['accumulation']==2 and c['effective_batch']==64
records={}
for rank in (0,1):
 rows=[json.loads(x) for x in (p/f'rank{rank}.jsonl').read_text().splitlines()]
 assert rows and rows[0]['phase']=='fake_D' and rows[0]['fake_update']==11, 'resume has not completed its first update'
 assert len(rows[0]['original_indices_rank'])==32
 assert all(math.isfinite(v) for row in rows for v in row.values() if isinstance(v,float))
 records[str(rank)]=rows
mon=[json.loads(x) for x in (root/'resource_monitor.jsonl').read_text().splitlines()]
assert all(not x['foreign'] for x in mon)
log=Path('logs/joint4_uniform_10k_2gpu_mb16_20261023_train.log').read_text()
assert 'RESUMED 0' in log and 'RESUMED 1' in log
out={'recorded_unix':time.time(),'run_directory':'/data5/shenglin.zeng/sdxl_crossdistill/'+str(p),
 'config':c,'status_at_capture':json.loads((p/'status.json').read_text()),
 'rows_at_capture':records,'restored_progress':{'gcount':2,'fcount':10,'dcount':10},
 'latest_resources':mon[-1],'source_sha256':json.loads((p/'source_sha256.json').read_text())}
(root/'resume_record.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'run':out['run_directory'],'restored_progress':out['restored_progress'],'status':out['status_at_capture']},indent=2))
PY
