import os,sys,time,json,csv,subprocess
from pathlib import Path
sys.path.insert(0,'/root/sdxl_research/code/sdxl_research')
from turning_batch64_workflow import descendants
root=Path('/root/autodl-tmp/sdxl_runs/outputs/a800_B_offload_compare_20261101')
with (root/'resources.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['unix_time','trial','process_tree_rss_GiB','gpu_used_MiB','gpu_free_MiB','gpu_util_percent'])
 while Path('/proc/3382/cmdline').exists():
  if b'a800_offload_compare.py' not in Path('/proc/3382/cmdline').read_bytes():break
  rss=0
  for pid in descendants(3382):
   try:
    for line in Path(f'/proc/{pid}/status').read_text().splitlines():
     if line.startswith('VmRSS:'):rss+=int(line.split()[1])*1024
   except OSError:pass
  logs=sorted(root.glob('*.log'),key=lambda p:p.stat().st_mtime)
  vals=subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
  w.writerow([time.time(),logs[-1].stem if logs else '',rss/1024**3,*[v.strip() for v in vals]]);f.flush();time.sleep(5)
