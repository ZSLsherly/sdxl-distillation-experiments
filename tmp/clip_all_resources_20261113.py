from clip_all_remote_20261113 import connect
from concurrent.futures import ThreadPoolExecutor
def check(h):
 try:
  c=connect(h);_,o,_=c.exec_command('nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader',timeout=30);o.channel.set_combine_stderr(True)
  s=o.read().decode();c.close();return h,s
 except Exception as e:return h,type(e).__name__
with ThreadPoolExecutor(max_workers=4) as ex:
 for h,s in ex.map(check,['172.18.32.78','172.18.33.60','172.18.32.81','172.18.33.9']):print(h,s,flush=True)
