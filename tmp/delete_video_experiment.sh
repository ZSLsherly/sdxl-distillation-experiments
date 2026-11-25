set -e
source /data5/shenglin.zeng/env.sh
python - <<'PY'
import json,shutil,subprocess,os
from pathlib import Path
from datetime import datetime
root=Path('/data5/shenglin.zeng').resolve()
relative=[
 'outputs/sparkdiffusion/CrossDistill_Smoke',
 'models/Wan2.1-T2V-1.3B',
 'models/negative_embed.pt',
 'cache/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B',
]
targets=[root/p for p in relative]
for p in targets:
 assert p.exists() and not p.is_symlink() and p.resolve().is_relative_to(root)
 assert not p.resolve().is_relative_to(root/'sdxl_crossdistill')
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if args and any(s in args[0] for s in ['python','torchrun']):
   assert not any(str(t) in a for a in args for t in targets),('Video path in use',p.name)
 except (FileNotFoundError,PermissionError):pass
items=[{'path':str(p),'allocated_bytes':int(subprocess.check_output(['du','-sx','--block-size=1',str(p)],text=True).split()[0])} for p in targets]
record={'created_at':datetime.now().astimezone().isoformat(),'reason':'User requested removal of old video CrossDistill experiment and associated video model','targets':items,'free_bytes_before':shutil.disk_usage(root).free}
dest=root/'sdxl_crossdistill/outputs/reviews/video_cleanup_20261018.json'
dest.write_text(json.dumps(record,indent=2))
print('TARGETS',json.dumps(items),flush=True)
for p in targets:
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
 print('REMOVED',str(p),flush=True)
assert all(not p.exists() for p in targets)
record['completed_at']=datetime.now().astimezone().isoformat()
record['free_bytes_after']=shutil.disk_usage(root).free
record['own_allocated_bytes_after']=int(subprocess.check_output(['du','-sx','--block-size=1',str(root)],text=True).split()[0])
dest.write_text(json.dumps(record,indent=2))
print('RESULT',json.dumps({'removed_GiB':sum(x['allocated_bytes'] for x in items)/1024**3,'own_GiB_after':record['own_allocated_bytes_after']/1024**3,'free_GiB_after':record['free_bytes_after']/1024**3}),flush=True)
m=json.loads((root/'sdxl_crossdistill/outputs/active_experiment.json').read_text())
for r in m['runs']:print('ACTIVE_STATUS',(root/'sdxl_crossdistill/outputs'/r['run']/'status.json').read_text())
PY
