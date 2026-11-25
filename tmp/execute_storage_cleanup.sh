set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,shutil,subprocess
from pathlib import Path
from datetime import datetime
root=Path.cwd().resolve();plan_path=root/'outputs/reviews/storage_cleanup_20261018.json'
plan=json.loads(plan_path.read_text());assert plan['root']==str(root)
active=json.loads((root/'outputs/active_experiment.json').read_text())
active_runs={r['run'] for r in active['runs']};target_runs={e['run'] for e in plan['remove']}
assert not (active_runs & target_runs)
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if '--run' in args:assert args[args.index('--run')+1] not in target_runs, ('Target run is still active',proc.name)
 except (FileNotFoundError,PermissionError):pass
for e in plan['keep']:
 p=Path(e['path']);assert (p/'complete.json').is_file() and any(p.glob('*.pt'))
for e in plan['remove']:
 p=Path(e['path']);expected=(root/'outputs'/e['run']/'weights').resolve()
 assert p.is_dir() and not p.is_symlink() and p.resolve().parent==expected
 assert expected.is_relative_to(root/'outputs') and (p/'complete.json').is_file()
 assert not any(f.is_symlink() for f in p.rglob('*'))
for i,e in enumerate(plan['remove'],1):
 shutil.rmtree(Path(e['path']))
 if i%20==0 or i==len(plan['remove']):print('REMOVED',i,'OF',len(plan['remove']),flush=True)
for e in plan['keep']:assert (Path(e['path'])/'complete.json').is_file() and any(Path(e['path']).glob('*.pt'))
plan['executed_at']=datetime.now().astimezone().isoformat()
plan['disk_after']=shutil.disk_usage(root)._asdict()
plan['own_allocated_bytes_after']=int(subprocess.check_output(['du','-sx','--block-size=1','/data5/shenglin.zeng'],text=True).split()[0])
plan_path.write_text(json.dumps(plan,indent=2))
print('RESULT',json.dumps({'deleted_checkpoints':len(plan['remove']),'retained_checkpoints':len(plan['keep']),'deleted_GiB':plan['delete_bytes']/1024**3,'free_GiB_before':plan['disk_before']['free']/1024**3,'free_GiB_after':plan['disk_after']['free']/1024**3,'own_GiB_before':plan['own_allocated_bytes_before']/1024**3,'own_GiB_after':plan['own_allocated_bytes_after']/1024**3}),flush=True)
for r in active['runs']:print('ACTIVE_STATUS',(root/'outputs'/r['run']/'status.json').read_text(),flush=True)
PY
df -h /data5
