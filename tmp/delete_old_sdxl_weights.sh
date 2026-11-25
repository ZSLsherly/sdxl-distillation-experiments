set -e
source /data5/shenglin.zeng/env.sh
python - <<'PY'
import json,re,shutil,subprocess
from pathlib import Path
from datetime import datetime
root=Path('/data5/shenglin.zeng/sdxl_crossdistill').resolve();out=root/'outputs'
groups=['hparam_pilot','gan_stage_pilot','gan_stage_pilot_v2','gan_stage_v3','gan_stage_v4','smoke_1024','v4_validation','state_tests']
protected=[out/'gan_stage_parallel',out/'gan_stage_joint34',out/'dmd_control']
active=json.loads((out/'active_experiment.json').read_text())
assert all((out/r['run']).resolve().is_relative_to(out/'dmd_control') for r in active['runs'])
def signature(p):
 s=p.stat();return (s.st_size,s.st_mtime_ns)
protected_weights={str(f):signature(f) for parent in protected for f in parent.rglob('*.pt') if f.is_file() and 'weights' in f.parts and (f.parent/'complete.json').exists()}
targets=[];preserved={};summary=[]
for group in groups:
 parent=out/group;count=0;total=0
 for p in sorted(parent.rglob('*')):
  if not p.is_file():continue
  is_weight=(bool(re.fullmatch(r'student_rank\d+\.pt',p.name)) or p.name in ['discriminator.pt','diffusion_pytorch_model.safetensors'] or (group=='state_tests' and bool(re.fullmatch(r'rank\d+\.pt',p.name))))
  if not is_weight:
   preserved[str(p)]=signature(p);continue
  assert not p.is_symlink() and p.resolve().is_relative_to(parent.resolve())
  assert not any(p.resolve().is_relative_to(q.resolve()) for q in protected)
  e={'path':str(p),'group':group,'bytes':p.stat().st_size};targets.append(e);count+=1;total+=e['bytes']
 summary.append({'group':group,'files':count,'GiB':total/1024**3})
assert targets
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if '--run' in args:
   run=args[args.index('--run')+1];assert not any(run==g or run.startswith(g+'/') for g in groups)
 except (FileNotFoundError,PermissionError):pass
record={'created_at':datetime.now().astimezone().isoformat(),'reason':'User requested deletion of early SDXL tuning/preflight/smoke weights only; preserve pictures and logs; exclude recent three groups and active DMD','targets':targets,'summary':summary,'protected_weight_file_count':len(protected_weights),'preserved_other_file_count':len(preserved),'free_before':shutil.disk_usage(root).free}
dest=out/'reviews/old_sdxl_weight_cleanup_20261018.json';dest.write_text(json.dumps(record,indent=2))
print('PLAN',json.dumps(summary),'TOTAL_GiB',sum(e['bytes'] for e in targets)/1024**3,flush=True)
for e in targets:
 p=Path(e['path']);assert p.stat().st_size==e['bytes'];p.unlink()
assert all(not Path(e['path']).exists() for e in targets)
assert all(Path(p).is_file() and signature(Path(p))==sig for p,sig in preserved.items())
assert all(Path(p).is_file() and signature(Path(p))==sig for p,sig in protected_weights.items())
record['completed_at']=datetime.now().astimezone().isoformat();record['free_after']=shutil.disk_usage(root).free
record['own_allocated_bytes_after']=int(subprocess.check_output(['du','-sx','--block-size=1','/data5/shenglin.zeng'],text=True).split()[0])
dest.write_text(json.dumps(record,indent=2))
print('RESULT',json.dumps({'deleted_weight_files':len(targets),'deleted_GiB':sum(e['bytes'] for e in targets)/1024**3,'own_GiB_after':record['own_allocated_bytes_after']/1024**3,'free_GiB_after':record['free_after']/1024**3,'preserved_other_files_verified':len(preserved),'protected_weights_verified':len(protected_weights)}),flush=True)
for run in active['runs']:print('ACTIVE_STATUS',(out/run['run']/'status.json').read_text(),flush=True)
PY
