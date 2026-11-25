"""Upload committed download chunks immediately, then publish only after remote SHA256."""
import json,queue,threading,time,math
from pathlib import Path
import upload_parallel as upl
from state_io import atomic_json,read_json
B=Path(__file__).resolve().parent
FILES=[('Wan2.1_VAE.pth',507609880,'38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981'),('backbone.pth',28578152777,'569e41d3046fe8bdd8df8345a9715e2d8fc5d92d52811ac165ad8947d776a47b'),('models_t5_umt5-xxl-enc-bf16.pth',11361920418,'7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d')]
CHUNK=4*1024**2

def run(name,size,sha):
 final=B/'weights'/name;partial=B/'weights'/(name+'.partial')
 ready_file=B/'weights'/(name+'.chunks.json');state=B/(name+'.uploaded.json')
 dest=upl.remote+'official/pf_wan/checkpoints/PF_Wan/'+name
 c=upl.connect();s=c.open_sftp()
 try: exists=s.stat(dest).st_size==size
 except IOError:exists=False
 if exists:
  _,o,_=c.exec_command("sha256sum '"+dest+"'",timeout=1800)
  assert o.read().decode().split()[0]==sha
  s.close();c.close();print('ALREADY_REMOTE_VERIFIED',name,flush=True);return
 done=set(read_json(state)) if state.exists() else set()
 try:assert s.stat(dest+'.parallel').st_size==size
 except IOError:
  with s.open(dest+'.parallel','w') as f:f.truncate(size)
  done=set()
 s.close();c.close()
 q=queue.Queue();stop=threading.Event();lock=threading.Lock();errors=[];queued=set(done)
 t0=time.time();bytes_sent=[0];last_print=[t0];total=math.ceil(size/CHUNK)
 def worker():
  c=None
  try:
   while not stop.is_set():
    try:a=q.get(timeout=3)
    except queue.Empty:continue
    path=final if final.exists() else partial
    try:
     with path.open('rb') as f:f.seek(a);buf=f.read(min(CHUNK,size-a))
    except FileNotFoundError:
     with final.open('rb') as f:f.seek(a);buf=f.read(min(CHUNK,size-a))
    assert len(buf)==min(CHUNK,size-a)
    for attempt in range(5):
     try:
      if c is None:
       c=upl.connect();c.get_transport().set_keepalive(20);s=c.open_sftp();s.get_channel().settimeout(90)
      with s.open(dest+'.parallel','r+') as g:g.set_pipelined(True);g.seek(a);g.write(buf)
      break
     except (EOFError,OSError,upl.paramiko.SSHException) as e:
      if c:c.close()
      c=None
      print('SSH_CHUNK_RECONNECT',name,a,type(e).__name__,'attempt',attempt+1,flush=True)
      if attempt==4:raise
      if stop.wait(min(2**attempt,10)):return
    with lock:
     done.add(a);bytes_sent[0]+=len(buf)
     atomic_json(state,sorted(done))
     if time.time()-last_print[0]>=10:
      print('STREAM_UPLOAD',name,'MiB',len(done)*4,'of',round(size/1024**2),'speed',round(bytes_sent[0]/(time.time()-t0)/1024**2,2),flush=True)
      last_print[0]=time.time()
  except Exception as e:errors.append(e);stop.set()
  finally:
   if c:c.close()
 threads=[]
 for _ in range(8):
  t=threading.Thread(target=worker);t.start();threads.append(t);time.sleep(.15)
 try:
  while not stop.is_set():
   if final.exists():ready=set(range(0,size,CHUNK))
   elif ready_file.exists():ready=set(read_json(ready_file))
   else:ready=set()
   for a in sorted(ready-queued):queued.add(a);q.put(a)
   with lock:complete=len(done)==total
   if complete and final.exists():break
   time.sleep(2)
 finally:
  stop.set()
  for t in threads:t.join()
 if errors:raise errors[0]
 assert len(done)==total
 c=upl.connect();_,o,_=c.exec_command("sha256sum '"+dest+".parallel'",timeout=1800)
 actual=o.read().decode().split()[0];assert actual==sha,(name,actual)
 s=c.open_sftp();s.posix_rename(dest+'.parallel',dest);s.close();c.close()
 print('REMOTE_VERIFIED',name,sha,flush=True)

if __name__=='__main__':
 for item in FILES:run(*item)
