from archive_results_n9 import cloud,lab,run,q,root,N,digest,clean,note
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import PurePosixPath
import time
rel='outputs/A_pcm10_mix1_g160/model/weights/20261106T190544+0800_g000160_end/student_rank0.pt'
src=root+rel; dst=N+'/'+rel; part=dst+'.partial'; block=8<<20
# Keep incoming tail blocks separate from the previously received prefix.
with cloud() as s,lab() as d:
 size=int(run(s,'stat -c %s '+q(src))); assert size==5135506134
 h=digest(s,src)
 if run(d,'test -f '+q(dst)+' && echo ready || true')=='ready':
  assert digest(d,dst)==h;clean(s,src,h,dst);raise SystemExit
 n=int(run(d,'stat -c %s '+q(part)));assert n<=size
 assert run(s,'head -c '+str(n)+' -- '+q(src)+' | sha256sum').split()[0]==digest(d,part)
 print('Verified prefix',n,'remaining',size-n,flush=True)
 def get(off):
  count=min(block,size-off);target=part+'.block.'+str(off)
  for attempt in range(4):
   try:
    with cloud() as a,lab() as b:
     if run(b,'test -f '+q(target)+' && stat -c %s '+q(target)+' || true')==str(count):return target
     _,rd,_=a.exec_command('dd if='+q(src)+' iflag=skip_bytes,count_bytes skip='+str(off)+' count='+str(count)+' bs=1M status=none',timeout=180)
     payload=rd.read();assert len(payload)==count and rd.channel.recv_exit_status()==0
     w,ack,_=b.exec_command('umask 077; cat > '+q(target+'.tmp')+' && mv -T -- '+q(target+'.tmp')+' '+q(target),timeout=180)
     w.channel.sendall(payload);w.channel.shutdown_write();assert ack.channel.recv_exit_status()==0
     return target
   except Exception:
    if attempt==3:raise
    time.sleep(2)
 offsets=list(range(n,size,block)); done=0
 with ThreadPoolExecutor(8) as pool:
  for fut in as_completed([pool.submit(get,x) for x in offsets]):
   fut.result();done+=1
   if done%16==0:print('Received tail blocks',done,'/',len(offsets),flush=True)
 assert int(run(d,'stat -c %s '+q(part)))==n
 blocks=[part+'.block.'+str(x) for x in offsets]
 if blocks:run(d,'cat -- '+' '.join(map(q,blocks))+' >> '+q(part))
 assert int(run(d,'stat -c %s '+q(part)))==size and digest(d,part)==h
 run(d,'mv -T -- '+q(part)+' '+q(dst))
 assert digest(d,dst)==h
 clean(s,src,h,dst)
 if blocks:run(d,'rm -- '+' '.join(map(q,blocks)))
 print('SOURCE_ABSENT',run(s,'test ! -e '+q(src)+' && echo yes'),run(s,'df -h /root/autodl-tmp'),flush=True)
