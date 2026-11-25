"""Bounded local preparation and result collection for the authorized 8-sample run."""
import json,subprocess,sys,time,traceback
import os,ctypes
from pathlib import Path
import upload_parallel as upl
from state_io import atomic_json,read_json
B=Path(__file__).resolve().parent
deadline=time.monotonic()+48*3600
procs=[]
def spawn(script,args,log):
 f=(B/log).open('a')
 p=subprocess.Popen([sys.executable,str(B/script),*args],stdout=f,stderr=subprocess.STDOUT)
 procs.append((p,f));return p
def state(stage,**kw):
 atomic_json(B/'local_status.json',{'stage':stage,'time_unix':time.time(),**kw})
def remote_status():
 c=upl.connect();s=c.open_sftp()
 try:
  with s.open(upl.remote+'status.json') as f:r=json.loads(f.read())
 finally:s.close();c.close()
 atomic_json(B/'server_status.json',r);return r
try:
 if os.name=='nt':
  assert ctypes.windll.kernel32.SetThreadExecutionState(0x80000001),'Unable to keep system awake during the authorized run'
 state('downloading_and_uploading')
 dl=spawn('download_local.py',[],'download_local.log')
 up=spawn('upload_stream.py',[],'upload_models.log')
 retries={'download':0,'upload':0}
 last_download_count=-1;last_download_progress=time.monotonic()
 recovery_count=0
 while time.monotonic()<deadline:
  counts=sum(len(read_json(p)) for p in (B/'weights').glob('*.chunks.json'))
  if counts!=last_download_count:
   last_download_count=counts;last_download_progress=time.monotonic()
  if counts-recovery_count>=32:
   # Five retries means consecutive failures without substantive new data.
   retries['download']=0;recovery_count=counts
  if dl.poll() is None and time.monotonic()-last_download_progress>300:
   # Terminate only the exact download child spawned by this controller.
   print('DOWNLOAD_WATCHDOG_RESTART',counts,flush=True)
   dl.terminate();dl.wait(timeout=30);last_download_progress=time.monotonic()
  if dl.poll() not in [None,0]:
   retries['download']+=1
   recovery_count=counts
   if retries['download']>5:raise RuntimeError('Model download failed after retries; see download_local.log')
   time.sleep(15);dl=spawn('download_local.py',[],'download_local.log');last_download_progress=time.monotonic()
  if up.poll() not in [None,0]:
   retries['upload']+=1
   if retries['upload']>5:raise RuntimeError('Model upload failed after retries; see upload_models.log')
   time.sleep(15);up=spawn('upload_stream.py',[],'upload_models.log')
  try:r=remote_status()
  except Exception as e:
   state('connection_retry',error=type(e).__name__);time.sleep(30);continue
  state('working',server=r)
  if r['stage']=='failed':
   state('server_failed',server=r)
   # Preparation is useful for recovery; retain active downloads, collect diagnostics.
   c=upl.connect();s=c.open_sftp()
   for name in ['controller.log','install_fast.log','run.log','environment.txt']:
    try:s.get(upl.remote+name,str(B/name))
    except IOError:pass
   s.close();c.close()
   raise RuntimeError(r['error'])
  if r['stage']=='complete':
   c=upl.connect();s=c.open_sftp()
   def fetch_dir(remote,local):
    import stat
    local.mkdir(exist_ok=True,parents=True)
    for item in s.listdir_attr(remote):
     if stat.S_ISDIR(item.st_mode):fetch_dir(remote+'/'+item.filename,local/item.filename)
     else:s.get(remote+'/'+item.filename,str(local/item.filename))
   for name in ['analysis','profiles','videos']:fetch_dir(upl.remote+name,B/name)
   for name in ['environment.txt','protocol.json','status.json','run.log']:s.get(upl.remote+name,str(B/name))
   s.close();c.close();state('complete',trajectories=r['trajectories']);break
  time.sleep(30)
 else:raise TimeoutError('48-hour preparation/run deadline exceeded')
except Exception as e:
 state('failed',error=str(e));traceback.print_exc();raise
finally:
 if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
 for p,f in procs:f.close()
