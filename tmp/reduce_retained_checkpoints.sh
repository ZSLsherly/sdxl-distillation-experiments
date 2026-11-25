set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,shutil,subprocess
from pathlib import Path
from datetime import datetime
root=Path.cwd().resolve();out=root/'outputs'
selected=['gan_stage_parallel/joint4_20261017T232317_mb16','gan_stage_parallel/gan4_20261017T232317_mb16','gan_stage_joint34/joint34_20261018T091442_4gpu_mb16']
active=json.loads((out/'active_experiment.json').read_text());assert not ({r['run'] for r in active['runs']} & set(selected))
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if '--run' in args:assert args[args.index('--run')+1] not in selected
 except (FileNotFoundError,PermissionError):pass
plan={'created_at':datetime.now().astimezone().isoformat(),'keep':[],'remove':[],'runs':[],'free_before':shutil.disk_usage(root).free}
for run in selected:
 parent=(out/run/'weights').resolve();entries=[]
 for p in sorted(parent.iterdir()):
  if not p.is_dir():continue
  assert not p.is_symlink() and p.resolve().parent==parent
  marker=p/'complete.json';assert marker.is_file()
  meta=json.loads(marker.read_text());files=list(p.rglob('*'))
  assert all(not f.is_symlink() for f in files)
  assert all(f.suffix in ('.pt','.json') for f in files if f.is_file())
  entries.append({'path':str(p),'g':meta['student_updates'],'bytes':sum(f.stat().st_size for f in files if f.is_file())})
 latest=max(e['g'] for e in entries);retain={32,64,118,latest}
 assert retain <= {e['g'] for e in entries}
 for e in entries:plan['keep' if e['g'] in retain else 'remove'].append(e)
 plan['runs'].append({'run':run,'keep_steps':sorted(retain),'before_count':len(entries),'after_count':sum(e['g'] in retain for e in entries)})
dest=out/'reviews/storage_cleanup_reduce_20261018.json';dest.write_text(json.dumps(plan,indent=2))
print('PLAN',json.dumps({'runs':plan['runs'],'delete_count':len(plan['remove']),'free_GiB':sum(e['bytes'] for e in plan['remove'])/1024**3}),flush=True)
for e in plan['remove']:
 p=Path(e['path']);assert p.resolve().is_relative_to(out) and p.parent.name=='weights'
 shutil.rmtree(p)
for e in plan['keep']:assert (Path(e['path'])/'complete.json').is_file() and any(Path(e['path']).glob('*.pt'))
plan['completed_at']=datetime.now().astimezone().isoformat();plan['free_after']=shutil.disk_usage(root).free
plan['own_allocated_bytes_after']=int(subprocess.check_output(['du','-sx','--block-size=1','/data5/shenglin.zeng'],text=True).split()[0])
dest.write_text(json.dumps(plan,indent=2))
print('RESULT',json.dumps({'deleted_GiB':sum(e['bytes'] for e in plan['remove'])/1024**3,'retained_count':len(plan['keep']),'own_GiB_after':plan['own_allocated_bytes_after']/1024**3,'free_GiB_after':plan['free_after']/1024**3}),flush=True)
for r in active['runs']:print('ACTIVE_STATUS',(out/r['run']/'status.json').read_text())
PY
du -x -B1 --max-depth=2 outputs/hparam_pilot outputs/gan_stage_pilot_v2 outputs/smoke_1024 outputs/gan_stage_v3 outputs/gan_stage_v4 /data5/shenglin.zeng/outputs/sparkdiffusion/CrossDistill_Smoke/wan1.3B_dense_pcm_dmd_smoke/checkpoints
