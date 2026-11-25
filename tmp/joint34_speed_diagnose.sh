source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,statistics,hashlib
from pathlib import Path
roots=[Path('outputs/gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16'),Path('outputs/gan_stage_joint34_4gpu/joint34_20261019T195324_4gpu_mb16')]
configs=[]
for p in roots:
 print('RUN',p)
 c=json.loads((p/'config.json').read_text());configs.append(c)
 print('HASHES',(p/'source_sha256.json').read_text())
 rows=[json.loads(x) for x in (p/'rank0.jsonl').read_text().splitlines()]
 gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase']=='fake_D']
 print('LAST_STATUS',(p/'status.json').read_text())
 for label,rr in [('FIRST_G',gs[:6]),('LAST_G',gs[-6:]),('FIRST_FAKE',fs[:10]),('LAST_FAKE',fs[-10:])]:
  print(label,[{k:r.get(k) for k in ['g','fake_update','stage','seconds','cycle_seconds','timestamp']} for r in rr])
 for stage in [2,3,4]:
  a=[r for r in gs[-12:] if r['stage']==stage]
  if a:print('RECENT_STAGE_MEAN',stage,'n',len(a),'Gseconds',statistics.mean(r['seconds'] for r in a),'cycle',statistics.mean(r['cycle_seconds'] for r in a))
print('CONFIG_DIFF',{k:[configs[0].get(k),configs[1].get(k)] for k in configs[0].keys()|configs[1].keys() if configs[0].get(k)!=configs[1].get(k)})
PY
nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,pstate --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
nvidia-smi pmon -c 5 -d 1
ps -eo user,pid,ppid,pcpu,pmem,etimes,args --sort=-pcpu | head -n 18
vmstat 1 3
