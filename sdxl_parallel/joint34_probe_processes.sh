source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import subprocess,json
from pathlib import Path
m=json.loads(Path('outputs/gan_stage_joint34/probe_launch.json').read_text())
pids=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  a=(p/'cmdline').read_bytes().decode().split('\0')
  if '--run' in a and a[a.index('--run')+1]==m['run']:pids.append(p.name)
 except (FileNotFoundError,PermissionError,ValueError):pass
if pids:subprocess.run(['ps','-p',','.join(pids),'-o','pid,ppid,stat,etime,time,%cpu,%mem,rss,wchan:24,args'])
print('LOG_BYTES',Path(m['log']).stat().st_size)
PY
