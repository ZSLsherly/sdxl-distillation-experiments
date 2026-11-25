from pathlib import Path
import os,sys,tarfile,hashlib,json,subprocess,time,concurrent.futures
os.umask(0o077)
dir=Path(__file__).resolve().parent; target=dir.parent
ssh=['ssh','-T','-oBatchMode=yes','-oIdentitiesOnly=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=15','-oServerAliveInterval=15','-oServerAliveCountMax=4','-oUserKnownHostsFile='+str(dir/'hosts'),'-i',str(dir/'key'),'-p','14892','root@connect.nma1.seetacloud.com']
m=json.loads(subprocess.check_output(ssh+['manifest'],timeout=100))
(target/'manifest.json').write_text(json.dumps(m,indent=1))
def digest(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def check(p,rec):
 assert p.stat().st_size==rec[0]
 with p.open('rb') as f:assert digest(f)==rec[1],p
def call(cmd,sink):
 for tries in range(4):
  begin=sink.tell()
  if subprocess.run(ssh+[cmd],stdout=sink).returncode==0:return
  sink.seek(begin);time.sleep(2**tries)
 raise RuntimeError(cmd)
with concurrent.futures.ThreadPoolExecutor(8) as ex:
 for i,rel in sorted(enumerate(m['large']),key=lambda x:('student_rank0' not in x[1],x[1])):
  rec=m['records'][rel];p=target/rel;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():
   tmp=p.with_suffix('.direct-part');tmp.touch()
   with tmp.open('r+b') as f:f.truncate(rec[0])
   def part(off):
    n=min(64<<20,rec[0]-off)
    with tmp.open('r+b',buffering=0) as f:
     f.seek(off);call(f'b {i} {off} {n}',f);assert f.tell()==off+n
   for _ in ex.map(part,range(0,rec[0],64<<20)):pass
   check(tmp,rec);os.link(tmp,p);tmp.unlink()
  else:check(p,rec)
  with (target/'verified_large.txt').open('a') as f:f.write(f'{rec[1]}  {rel}\n')
  print('VERIFIED',rel,flush=True)
 def group(entry):
  i,rels=entry;folder=target/'evidence';folder.mkdir(exist_ok=True)
  p=folder/f'part_{i:03d}.tar';tmp=p.with_suffix('.partial')
  if not p.exists():
   with tmp.open('wb') as f:call(f'g {i}',f)
  with tarfile.open(p if p.exists() else tmp,mode='r|') as t:
   seen=[]
   for a in t:
    assert a.isfile() and a.name in m['records']
    size,h=m['records'][a.name]
    assert a.size==size and digest(t.extractfile(a))==h,a.name
    seen.append(a.name)
   assert seen==rels
  if not p.exists():tmp.rename(p)
 for _ in ex.map(group,enumerate(m['groups'])):pass
(target/'COMPLETE.json').write_text(json.dumps({'files':len(m['records']),'bytes':sum(r[0] for r in m['records'].values()),'large':len(m['large']),'tar_groups':len(m['groups'])}))
print('COMPLETE',flush=True)

