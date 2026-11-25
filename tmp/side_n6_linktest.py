from backup_a800_nov5 import cloud,lab,run
import time,concurrent.futures
def source():
 with cloud() as c:
  st=time.monotonic();_,r,e=c.exec_command('dd if=/dev/urandom bs=1M count=4 status=none',timeout=60)
  n=0
  while b:=r.read(1<<20):n+=len(b)
  assert r.channel.recv_exit_status()==0
  return ['cloud_to_local_memory',n,round(time.monotonic()-st,2)]
def dest():
 with lab() as c:
  st=time.monotonic();w,r,e=c.exec_command('cat >/dev/null',timeout=60)
  for i in range(4):w.channel.sendall(b'x'*(1<<20))
  w.channel.shutdown_write();assert r.channel.recv_exit_status()==0
  return ['local_to_lab_discard',4<<20,round(time.monotonic()-st,2)]
with concurrent.futures.ThreadPoolExecutor(2) as e:
 for r in e.map(lambda f:f(),[source,dest]):print(r,flush=True)
