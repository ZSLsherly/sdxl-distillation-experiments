from archive_results_n9 import cloud,lab,run,q,N,root
from pathlib import PurePosixPath
import hashlib
model='outputs/A_pcm10_mix1_g160/model/'
out=model+'weights/20261106T190544+0800_g000160_end/'
with cloud() as s,lab() as d,s.open_sftp() as a,d.open_sftp() as b:
 for rel in [out+'complete.json',out+'shapes_rank0.json',model+'config.json',model+'status.json',model+'data_manifest.json']:
  data=a.open(root+rel).read();h=hashlib.sha256(data).hexdigest();dst=N+'/'+rel
  run(d,'mkdir -p -- '+q(str(PurePosixPath(dst).parent)))
  if not run(d,'test ! -f '+q(dst)+' || echo exists'):
   with b.open(dst+'.meta-tmp','wb') as w:w.write(data)
   b.rename(dst+'.meta-tmp',dst)
  assert run(d,'sha256sum -- '+q(dst)).split()[0]==h
  print('verified',rel,flush=True)
