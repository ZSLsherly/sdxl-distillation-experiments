"""Refresh local snapshot of this supplement; never copies model weights locally."""
from pathlib import Path
import json
from clip_all_remote_20261113 import connect
r='/data3/shenglin.zeng/clip_missing_20261113';out=Path('output/sdxl-crossdistill/clip_all_20261113')
c=connect('172.18.32.81')
with c.open_sftp() as s:
 for name in ('report.html','results_all.json','评测记录.txt','validation.json','queue_5.log'):
  try:s.get(r+'/'+name,str(out/name))
  except FileNotFoundError:pass
 states={}
 try:arms=s.listdir(r+'/arms')
 except FileNotFoundError:arms=[]
 for arm in arms:
  state={}
  for name in ('ready.json','status.json','results.json'):
   try:
    with s.open(r+'/arms/'+arm+'/'+name) as f:state[name]=json.load(f)
   except FileNotFoundError:pass
  states[arm]=state
 (out/'live_snapshot.json').write_text(json.dumps(states,ensure_ascii=False,indent=2),encoding='utf-8')
 print('READY',sum('ready.json' in v for v in states.values()),'NEW_COMPLETE',sum('results.json' in v for v in states.values()))
 for a,v in states.items():
  if 'status.json' in v:print(a,v['status.json'])
c.close()
