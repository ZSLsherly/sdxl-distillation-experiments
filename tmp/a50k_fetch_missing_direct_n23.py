"""Official public shards download on .60, then verified rclone copy to .81."""
import json, shlex, sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'tmp'))
from a50k_stage_assets_n23 import AUTH, OUT, DEST, remote, record, paramiko, upload
code=r'''
import json,sys,subprocess,os,hashlib,urllib.request,time,concurrent.futures,shutil
from pathlib import Path
r=json.loads(sys.stdin.readline())
rc='/data5/shenglin.zeng/checkpoints/a800_archive_20261114/.tools/rclone'
env=dict(os.environ,RCLONE_CONFIG='/dev/null',RCLONE_CONFIG_DEST_TYPE='sftp',
    RCLONE_CONFIG_DEST_HOST='172.18.32.81',RCLONE_CONFIG_DEST_PORT='22',RCLONE_CONFIG_DEST_USER=r['user'])
env['RCLONE_CONFIG_DEST_PASS']=subprocess.check_output([rc,'obscure','-'],input=r.pop('password')+'\n',universal_newlines=True).strip()
folder=Path('/data5/shenglin.zeng/datasets/a50k_sources_20261123');folder.mkdir(parents=True,exist_ok=True)
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def download_parallel(spec,path):
 chunks=folder/(spec['name']+'.chunks');chunks.mkdir(exist_ok=True)
 block=16*1024**2
 url=spec.get('direct_url',spec['url'].replace('https://huggingface.co','https://hf-mirror.com'))
 def fetch(start):
  end=min(start+block,spec['size'])-1;chunk=chunks/str(start)
  if chunk.exists() and chunk.stat().st_size==end-start+1:return chunk
  for attempt in range(8):
   try:
    req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=45) as response:
     assert response.status==206 and response.headers['Content-Range']==f'bytes {start}-{end}/{spec["size"]}'
     with chunk.with_suffix('.tmp').open('wb') as f:shutil.copyfileobj(response,f,1024**2)
    assert chunk.with_suffix('.tmp').stat().st_size==end-start+1
    chunk.with_suffix('.tmp').replace(chunk)
    print('CHUNK',spec['name'],start,end,flush=True);return chunk
   except Exception as e:
    status=getattr(e,'code',None)
    print('CHUNK_RETRY',spec['name'],start,type(e).__name__,status,flush=True)
    time.sleep(30 if status==429 else 5)
  raise RuntimeError('Chunk failed '+spec['name']+' '+str(start))
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
  files=list(pool.map(fetch,range(0,spec['size'],block)))
 part=path.with_suffix('.parallel_partial')
 with part.open('wb') as target:
  for chunk in files:
   with chunk.open('rb') as source:shutil.copyfileobj(source,target,8*1024**2)
 assert part.stat().st_size==spec['size'] and sha(part)==spec['sha256']
 part.replace(path)
 for chunk in files:chunk.unlink()
 chunks.rmdir()
for spec in r['specs']:
 path=folder/spec['name']
 for round_ in range(3):
  if path.exists():break
  try:download_parallel(spec,path)
  except RuntimeError:
   if round_==2:raise
   print('RESUME_INCOMPLETE_CHUNKS',spec['name'],flush=True);time.sleep(15)
 assert path.stat().st_size==spec['size'] and sha(path)==spec['sha256']
 print('OFFICIAL_SHA256_VERIFIED',spec['name'],flush=True)
 dst=r['dest']+'/datasets/sources/'+spec['name']
 subprocess.check_call([rc,'copyto',str(path),'dest:'+dst,'--stats','30s','--stats-one-line',
     '--log-level','NOTICE','--retries','2','--contimeout','15s'],env=env)
 with (folder/'下载与传输记录.txt').open('a') as f:
  f.write(json.dumps(dict(source_url=spec['url'],download_mirror='hf-mirror.com',
   source_host='172.18.33.60',source_path=str(path),destination_host='172.18.32.81',
   destination_path=dst,sha256=spec['sha256'],source_verified=True),ensure_ascii=False)+'\n')
 print('COPIED',spec['name'],flush=True)
'''
host='172.18.33.60'
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host,username=AUTH[host][0],password=AUTH[host][1],timeout=20,look_for_keys=False,allow_agent=False)
c.get_transport().set_keepalive(15)
specs=[s for s in json.loads((OUT/'sources.json').read_text())['files'] if 4<=s['shard']<=7]
i,o,e=c.exec_command('python3 -u -c '+shlex.quote(code),timeout=7200);o.channel.set_combine_stderr(True)
i.write(json.dumps(dict(user=AUTH['172.18.32.81'][0],password=AUTH['172.18.32.81'][1],specs=specs,dest=DEST))+'\n')
i.flush();i.channel.shutdown_write()
for line in o:print(line.rstrip(),flush=True)
assert o.channel.recv_exit_status()==0;c.close()
for spec in specs:
    dest=DEST+'/datasets/sources/'+spec['name']
    got=remote('sha256sum '+shlex.quote(dest)).split()[0]
    assert got==spec['sha256']
    record(dict(source_host='huggingface.co official via hf-mirror.com',source=spec['url'],
                staging_host=host,staging_path='/data5/shenglin.zeng/datasets/a50k_sources_20261123/'+spec['name'],
                destination_host='172.18.32.81',destination=dest,sha256=got,verified=True,method='server-side download and rclone'))
    print('DEST_HASH_VERIFIED',spec['name'],flush=True)
upload(OUT/'sources.json',DEST+'/datasets/sources/sources.json')
upload(OUT/'传输与下载校验记录.txt',DEST+'/records/transfer_verified.txt')
print('ALL_NEW_SHARDS_STAGED',flush=True)
