"""Finish this one evaluation's local report after the remote batch completes."""
from pathlib import Path
import json,subprocess,sys,time,os
root=Path(__file__).resolve().parents[1];os.chdir(root)
out=root/'output/sdxl-crossdistill/coco_uniform_g160_20261025'
deadline=time.monotonic()+6*3600
while time.monotonic()<deadline:
 result=subprocess.run([sys.executable,'tmp/fetch_coco_uniform_20261025.py'],capture_output=True,text=True,encoding='utf-8',errors='replace')
 if result.returncode:
  print('FETCH_RETRY',time.time(),result.stderr[-600:],flush=True)
 else:
  status=json.loads((out/'remote_status.json').read_text())
  done=(out/'complete.json').exists()
  events=status.get('guard_events',[])
  stopped=bool(events and events[-1]['event']=='stopped')
  if done:
   subprocess.run([sys.executable,'tmp/fetch_coco_uniform_20261025.py','--images'],check=True)
  subprocess.run([sys.executable,'tmp/write_coco_uniform_report_20261025.py'],check=True)
  n=sum(x['done'] for x in status['workers'].values())
  print('REPORT_UPDATED',time.time(),n,'complete',done,'stopped',stopped,flush=True)
  if done or stopped:break
 time.sleep(60)
else:
 print('COLLECTION_TIMEOUT: remote evaluation is not modified',flush=True)
