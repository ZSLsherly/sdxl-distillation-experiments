import os,sys,json,subprocess,selectors,time,hashlib
from pathlib import Path
c=json.loads(sys.stdin.readline());env=os.environ.copy()
base=Path('/data5/shenglin.zeng/checkpoints/a800_archive_20261106/.pull6')
env['HOME']=str(base/'fzhome');Path(env['HOME']).mkdir(exist_ok=True)
env['LD_LIBRARY_PATH']='/data5/shenglin.zeng/tools/filezilla/app/usr/lib/x86_64-linux-gnu'
p=subprocess.Popen(['/data5/shenglin.zeng/tools/filezilla/app/usr/bin/fzsftp'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,env=env)
def send(x):p.stdin.write((x+'\n').encode());p.stdin.flush()
send('-0-');send('-1-');send('open "root@connect.nma1.seetacloud.com" 14892')
sel=selectors.DefaultSelector();sel.register(p.stdout,selectors.EVENT_READ);buf=b'';hostline=False;verified=False;opened=False;start=time.monotonic()
try:
 while time.monotonic()-start<90:
  if not sel.select(2):continue
  b=os.read(p.stdout.fileno(),65536)
  if not b:break
  buf+=b
  while buf:
   if buf[:1] in (b'6',b'7',b'@',b'A') and not hostline:
    typ=buf[:1];buf=buf[1:]
    if typ==b'@':send('-0-')
    if typ==b'A':send('-1-')
    continue
   if b'\n' not in buf:break
   line,buf=buf.split(b'\n',1)
   if hostline:
    assert line.strip()==b'14892' and verified,'Unexpected SSH host identity'
    send('n');hostline=False;continue
   typ=line[:1];v=line[1:].decode(errors='replace')
   if typ==b'I':
    verified=c['sha256'] in v or c['md5'] in v
   if typ in (b'9',b':'):
    assert 'connect.nma1.seetacloud.com' in v
    hostline=True
   elif typ==b'<':
    assert verified;send(c['password'])
   elif typ==b'1':
    print('DONE',v,flush=True)
    if not opened:
     opened=True;send('get "/root/sdxl_research/tmp/side_archive_n6/manifest.json" "'+str(base/'fz_probe_manifest.json')+'"')
    else:
     f=base/'fz_probe_manifest.json';assert f.is_file()
     print('FILEZILLA_DOWNLOADED',f.stat().st_size,hashlib.sha256(f.read_bytes()).hexdigest(),flush=True);send('quit');sys.exit(0)
   elif typ not in (b'=',b'>',b'?'):
    print('EVENT',typ.decode(errors='replace'),v[:160],flush=True)
 raise RuntimeError('FileZilla probe did not complete')
finally:
 p.terminate();p.wait()
