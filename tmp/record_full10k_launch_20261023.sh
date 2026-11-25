set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,math,time
from pathlib import Path
root=Path('outputs/joint4_uniform_10k_2gpu_20261023');p=root/'train'
c=json.loads((p/'config.json').read_text())
assert not c['benchmark'] and c['train_count']==10000 and c['effective_batch']==64
assert c['accumulation']==8 and c['world_size']==2 and c['max_student_updates']==160
rows={str(rank):[json.loads(x) for x in (p/f'rank{rank}.jsonl').read_text().splitlines()] for rank in (0,1)}
for rank,entries in rows.items():
 gs=[x for x in entries if x['phase']=='student_done']
 assert gs and gs[0]['g']==1, 'first G update has not completed yet'
 for row in entries:
  assert len(row['original_indices_rank'])==32
  assert all(math.isfinite(v) for k,v in row.items() if isinstance(v,float)),(rank,row)
for a,b in zip(rows['0'],rows['1']):
 assert a['phase']==b['phase']
 assert len(set(a['original_indices_rank']+b['original_indices_rank']))==64
mon=[json.loads(x) for x in (root/'resource_monitor.jsonl').read_text().splitlines()]
assert all(not x['foreign'] for x in mon)
gs=[x for x in rows['0'] if x['phase']=='student_done']
result={'recorded_unix':time.time(),'server':'172.18.33.60','physical_gpus':[0,2],
 'run_directory':'/data5/shenglin.zeng/sdxl_crossdistill/'+str(p),
 'config':c,'status_at_capture':json.loads((p/'status.json').read_text()),
 'completed_G_updates_at_capture':gs,'latest_fake_updates':[x for x in rows['0'] if x['phase']=='fake_D'][-5:],
 'rough_160_update_compute_hours':gs[0]['cycle_seconds']*160/3600,
 'rough_eta_note':'single first-cycle extrapolation only; excludes later evaluations/checkpoints and load changes',
 'source_sha256':json.loads((p/'source_sha256.json').read_text()),
 'data_manifest':json.loads((p/'data_manifest.json').read_text()),
 'latest_resource_sample':mon[-1],'foreign_process_detected':False}
(root/'launch_record.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k in ('completed_G_updates_at_capture','rough_160_update_compute_hours','run_directory','status_at_capture')},indent=2))
PY
