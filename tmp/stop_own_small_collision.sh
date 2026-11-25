source /data5/shenglin.zeng/env.sh
python - <<'PY'
import os,signal
from pathlib import Path
for pid in (1897610,1897611):
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
  if 'sdxl_research/train.py' in cmd and 'research_small_20261022/bucket_offload_b8' in cmd:
   os.kill(pid,signal.SIGTERM);print('TERMINATED_OUR_WORKER',pid)
  else:print('NOT_MATCHING_OUR_WORKER',pid)
 except ProcessLookupError:print('ALREADY_EXITED',pid)
 except FileNotFoundError:print('ALREADY_EXITED',pid)
PY
nvidia-smi -i 0,3 --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
ps -p 1897507,1897610,1897611,1899365 -o pid,user,etime,comm
