"""Direct backup of two final weights ONLY, plus requested checksum records."""
from pathlib import Path, PurePosixPath
import concurrent.futures,datetime,hashlib,json,re,shlex,sys,time,threading,socket
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko

PW=sys.stdin.readline().strip()
LAB='/data5/shenglin.zeng/checkpoints/a800_archive_20261114'
ROOT='/root/sdxl_research'
STAGE=ROOT+'/tmp/backup_completed_side_20261114'
LOCAL=Path('output/sdxl-crossdistill/a800_archive_20261114_side')
LOCAL.mkdir(parents=True,exist_ok=True)
ARMS=['A_pcm10_terminal_low30_g160_20261113','A_clean_global_g160_20261113']
q=shlex.quote

def connect(host,port,user,pw):
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 sock=socket.create_connection((host,port),timeout=20)
 sock.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
 sock.setsockopt(socket.SOL_SOCKET,socket.SO_SNDBUF,1<<20)
 c.connect(host,port=port,username=user,password=pw,sock=sock,timeout=20,look_for_keys=False,allow_agent=False)
 c.get_transport().set_keepalive(30)
 return c

def run(c,cmd):
 _,o,_=c.exec_command(cmd,timeout=600);o.channel.set_combine_stderr(True)
 out=o.read().decode('utf8',errors='replace');rc=o.channel.recv_exit_status()
 if rc:raise RuntimeError(f'remote exit {rc}: '+out[-1600:].replace(PW,'[redacted]'))
 return out

def put_text(c,p,s):
 with c.open_sftp() as f:
  with f.file(p,'w') as out:out.write(s.encode('utf8'))

def direct_scp(c,origin,dest,label):
 # Native scp performs the data transfer directly from the laboratory server.
 cmd='scp -q -p -o StrictHostKeyChecking=yes -o UserKnownHostsFile='+q(LAB+'/.known_hosts')+' -o PreferredAuthentications=password -o PubkeyAuthentication=no -o NumberOfPasswordPrompts=1 -P 14892 '+q('root@connect.nma1.seetacloud.com:'+origin)+' '+q(dest)
 channel=c.get_transport().open_session();channel.get_pty();channel.exec_command(cmd)
 text='';sent=False;start=last=time.monotonic()
 with c.open_sftp() as sf:
  while True:
   if channel.recv_ready():
    text+=channel.recv(65536).decode('utf8',errors='replace')
    if not sent and 'password:' in text.lower():
     channel.sendall(PW+'\n');sent=True
   if channel.exit_status_ready() and not channel.recv_ready():break
   if time.monotonic()-last>=25:
    try:size=sf.stat(dest).st_size
    except OSError:size=0
    print('TRANSFER',label,round(size/2**30,3),'GiB',round(time.monotonic()-start),'seconds',flush=True)
    last=time.monotonic()
   if time.monotonic()-start>3600:
    channel.close();raise TimeoutError(label)
   time.sleep(.2)
 rc=channel.recv_exit_status();channel.close()
 if rc:raise RuntimeError('scp '+label+' failed: '+text.replace(PW,'[redacted]')[-800:])
 print('COPIED',label,round(time.monotonic()-start,1),'seconds',flush=True)

def main():
 raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
 a=re.search(r'IP:\s*172\.18\.33\.60\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)
 with connect('connect.nma1.seetacloud.com',14892,'root',PW) as source,connect('172.18.33.60',22,a[1],a[2].strip()) as lab:
  prepare=r'''
from pathlib import Path
import hashlib,json,os
r=Path(ROOT);stage=Path(STAGE);stage.mkdir(parents=True,exist_ok=True)
records={};weights=[];groups={}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def add(p):
 assert not p.is_symlink() and p.is_file(),p
 rel=str(p.relative_to(r));records[rel]={'bytes':p.stat().st_size,'sha256':digest(p)}
for arm in ARMS:
 job=r/'outputs'/arm;model=job/'model'
 state=json.loads((model/'status.json').read_text());done=json.loads((job/'complete.json').read_text())
 assert state['status']=='complete' and state['student_updates']==160 and state['fake_updates']==320
 assert done['G']==160 and done['fake']==320
 checkpoints=list((model/'weights').glob('*/student_rank0.pt'));assert len(checkpoints)==1
 ck=checkpoints[0];validation=json.loads((ck.parent/'cpu_validation.json').read_text())
 assert validation['strict_load'] and validation['finite']
 add(ck);weights.append(str(ck.relative_to(r)))
 groups[arm]={'G':160,'fake':320,'weight':str(ck.relative_to(r))}
manifest={'source_root':str(r),'groups':groups,'weights':weights,'records':records,
          'scope':'two final student weight files only; no configs/source/data/logs/images'}
(stage/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
'''.replace('ROOT',repr(ROOT)).replace('STAGE',repr(STAGE)).replace('ARMS',repr(ARMS))
  print('HASHING_COMPLETED_RUNS',flush=True)
  manifest=json.loads(run(source,"nice -n 10 /root/sdxl_research/env/bin/python - <<'PY'\n"+prepare+"\nPY"))
  (LOCAL/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
  print('MANIFEST',len(manifest['records']),'files',round(sum(v['bytes'] for v in manifest['records'].values())/2**30,3),'GiB',flush=True)
  run(lab,'umask 077; mkdir -p '+q(LAB))
  if run(lab,'test ! -e '+q(LAB+'/COMPLETE.json')+' || echo complete').strip():
   raise RuntimeError('Destination already complete: verify existing backup instead of replacing it')
  key=source.get_transport().get_remote_server_key()
  put_text(lab,LAB+'/.known_hosts',f'[connect.nma1.seetacloud.com]:14892 {key.get_name()} {key.get_base64()}\n')
  put_text(lab,LAB+'/manifest.json',json.dumps(manifest,indent=2))
  pending=[];resume={}
  for rel in manifest['weights']:
   rec=manifest['records'][rel];dest=LAB+'/'+rel
   run(lab,'mkdir -p '+q(str(PurePosixPath(dest).parent)))
   exists=run(lab,'test ! -f '+q(dest)+' || echo exists').strip()
   if exists:
    assert run(lab,'sha256sum '+q(dest)).split()[0]==rec['sha256']
   else:
    pending.append(rel)
    resume[rel]=0
    if run(lab,'test ! -f '+q(dest+'.partial')+' || echo partial').strip():
     for n in (1024<<20,512<<20,128<<20):
      if n>=rec['bytes']:continue
      prefix='dd bs=1M iflag=count_bytes count='+str(n)+' status=none if='
      h1=run(source,prefix+q(ROOT+'/'+rel)+' | sha256sum').split()[0]
      h2=run(lab,prefix+q(dest+'.partial')+' | sha256sum').split()[0]
      if h1==h2:
       resume[rel]=n;print('VERIFIED_RESUME_PREFIX',rel,n,flush=True);break
    run(lab,'truncate -s '+str(rec['bytes'])+' '+q(dest+'.partial'))
  tls=threading.local();connections=[];guard=threading.Lock();login_gate=threading.Semaphore(4)
  def transfer_chunk(task):
   rel,offset,size=task
   if not hasattr(tls,'source'):
    with login_gate:
     tls.source=connect('connect.nma1.seetacloud.com',14892,'root',PW)
     tls.dest=connect('172.18.33.60',22,a[1],a[2].strip())
    with guard:connections.extend([tls.source,tls.dest])
   for attempt in range(3):
    try:
     _,rd,err=tls.source.exec_command('dd if='+q(ROOT+'/'+rel)+' bs=1M iflag=skip_bytes,count_bytes,fullblock skip='+str(offset)+' count='+str(size)+' status=none',timeout=180)
     wr,ack,de=tls.dest.exec_command('dd of='+q(LAB+'/'+rel+'.partial')+' bs=1M iflag=fullblock oflag=seek_bytes conv=notrunc seek='+str(offset)+' status=none',timeout=180)
     count=0
     while data:=rd.read(1<<20):
      wr.channel.sendall(data);count+=len(data)
     wr.channel.shutdown_write()
     assert count==size and rd.channel.recv_exit_status()==0 and ack.channel.recv_exit_status()==0
     return rel,count
    except Exception:
     if attempt==2:raise
     time.sleep(2)
  tasks=[(rel,off,min(32<<20,manifest['records'][rel]['bytes']-off)) for rel in pending for off in range(resume[rel],manifest['records'][rel]['bytes'],32<<20)]
  start=last=time.monotonic();done=0;resumed=sum(resume.values())
  try:
   with concurrent.futures.ThreadPoolExecutor(8) as pool:
    for f in concurrent.futures.as_completed([pool.submit(transfer_chunk,t) for t in tasks]):
     rel,size=f.result();done+=size
     if time.monotonic()-last>20 or done+resumed==sum(manifest['records'][r]['bytes'] for r in pending):
      print('RAM_RELAY',round((done+resumed)/2**30,3),'GiB',round(done/2**20/(time.monotonic()-start),2),'MiB/s',flush=True);last=time.monotonic()
  finally:
   for c in connections:c.close()
  for rel in pending:
   rec=manifest['records'][rel];dest=LAB+'/'+rel
   assert run(lab,'sha256sum '+q(dest+'.partial')).split()[0]==rec['sha256']
   run(lab,'mv -n '+q(dest+'.partial')+' '+q(dest))
   print('WEIGHT_HASH_VERIFIED',rel,rec['sha256'],flush=True)
  verify=r'''
from pathlib import Path
import hashlib,json,datetime
r=Path(DEST);m=json.loads((r/'manifest.json').read_text())
assert set(m['records'])==set(m['weights']) and len(m['weights'])==2
for rel,rec in m['records'].items():
 p=r/rel;assert p.is_file() and p.stat().st_size==rec['bytes']
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==rec['sha256'],rel
result={'completed_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
 'destination':str(r),'verified_files':len(m['records']),'verified_bytes':sum(x['bytes'] for x in m['records'].values()),
 'groups':m['groups'],'weights':{k:m['records'][k] for k in m['weights']},'all_sha256_verified':True}
(r/'COMPLETE.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''.replace('DEST',repr(LAB))
  result=json.loads(run(lab,"nice -n 10 python3 - <<'PY'\n"+verify+"\nPY"))
  (LOCAL/'receipt.json').write_text(json.dumps(result,indent=2),encoding='utf8')
  lines=['A800已完成权重实验室备份记录',result['completed_at'],
         '目标服务器：172.18.33.60','目标目录：'+LAB,
         '原生SCP直传速度低，改为SSH分块内存转发；权重没有落盘到本机。',
         '本次备份仅包含两份最终学生权重和校验清单；配置、源码、数据清单、训练日志及样图不在本次传输中。',
         '本次只备份清单中已完成的两组；未删除A800源文件，未操作正在训练的A去中段组。',
         '逐文件SHA256验证：'+str(result['verified_files'])+'个文件全部一致。',
         '总字节数：'+str(result['verified_bytes']),
         '权重类型：BF16学生推理权重，不含fake或优化器续训状态。','']
  for rel,rec in result['weights'].items():
   lines.extend(['实验：'+rel.split('/')[1],'G160/F320','路径：'+LAB+'/'+rel,
                 '字节数：'+str(rec['bytes']),'SHA256：'+rec['sha256'],''])
  note='\n'.join(lines)+'\n'
  (LOCAL/'备份记录.txt').write_text(note,encoding='utf8')
  put_text(lab,LAB+'/备份记录.txt',note)
  put_text(source,STAGE+'/备份记录.txt',note)
  print('BACKUP_COMPLETE',result['verified_files'],result['verified_bytes'],flush=True)

if __name__=='__main__':main()
