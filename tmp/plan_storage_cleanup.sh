set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,shutil,subprocess
from pathlib import Path
from datetime import datetime
root=Path.cwd().resolve();out=root/'outputs'
selections={
 'gan_stage_parallel/joint4_20261017T232317_mb16':{4,32,64,96,114,118},
 'gan_stage_parallel/gan4_20261017T232317_mb16':{4,32,64,96,114,118,136,154,160},
 'gan_stage_joint34/joint34_20261018T091442_4gpu_mb16':{4,32,64,96,114,118,134,136,140},
 'gan_stage_v4/joint4_20261017T212010':{2},
 'gan_stage_v4/gan4_20261017T212010':{2},
}
active=json.loads((out/'active_experiment.json').read_text());active_runs={x['run'] for x in active['runs']}
plan={'created_at':datetime.now().astimezone().isoformat(),'root':str(root),'active_runs':sorted(active_runs),'runs':[],'remove':[],'keep':[],'disk_before':shutil.disk_usage(root)._asdict(),'own_allocated_bytes_before':int(subprocess.check_output(['du','-sx','--block-size=1','/data5/shenglin.zeng'],text=True).split()[0])}
for run,keep_steps in selections.items():
 assert run not in active_runs
 p=out/run
 entries=[]
 for q in sorted((p/'weights').iterdir()):
  if not q.is_dir():continue
  assert not q.is_symlink() and q.resolve().parent==(p/'weights').resolve()
  marker=q/'complete.json'
  if not marker.exists():continue
  meta=json.loads(marker.read_text())
  assert all(not f.is_symlink() for f in q.rglob('*'))
  files=[f for f in q.rglob('*') if f.is_file()]
  assert all(f.suffix in {'.pt','.json'} for f in files)
  entries.append({'path':str(q),'run':run,'g':meta['student_updates'],'bytes':sum(f.stat().st_size for f in files)})
 assert entries
 latest=max(e['g'] for e in entries);keep_steps.add(latest)
 for e in entries:plan['keep' if e['g'] in keep_steps else 'remove'].append(e)
 plan['runs'].append({'run':run,'before_count':len(entries),'before_GiB':sum(e['bytes'] for e in entries)/1024**3,'keep_steps':sorted({e['g'] for e in entries if e['g'] in keep_steps}),'delete_count':sum(e['g'] not in keep_steps for e in entries)})
plan['delete_bytes']=sum(e['bytes'] for e in plan['remove'])
dest=out/'reviews/storage_cleanup_20261018.json';dest.parent.mkdir(exist_ok=True)
dest.write_text(json.dumps(plan,indent=2))
print(json.dumps({'plan_path':str(dest),'delete_GiB':plan['delete_bytes']/1024**3,'delete_count':len(plan['remove']),'keep_count':len(plan['keep']),'runs':plan['runs'],'disk_free_GiB':plan['disk_before']['free']/1024**3,'own_GiB':plan['own_allocated_bytes_before']/1024**3},indent=2))
PY
