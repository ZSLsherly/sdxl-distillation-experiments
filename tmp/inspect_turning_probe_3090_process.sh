date -Is
ps -eo pid,ppid,etime,stat,pcpu,rss,args | grep -E 'run_shared_probe_retry1.py|capped_turning_train.py' | grep -v grep
python3 - <<'PY'
import subprocess
from pathlib import Path
for line in subprocess.check_output(['ps','-eo','pid,args'],universal_newlines=True).splitlines():
 if 'python -u /data1/shenglin.zeng/teacher_turning_20261026/outputs/turning_boundary_ablation_20261027/code/sdxl_research/capped_turning_train.py' in line:
  pid=line.split()[0]
  for name in ['wchan','io']:
   try:print(pid,name,(Path('/proc')/pid/name).read_text())
   except OSError:pass
PY
