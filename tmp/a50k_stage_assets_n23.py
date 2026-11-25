"""rclone-only public data staging; secrets stay in memory and environment."""
import argparse, concurrent.futures, hashlib, json, os, re, shlex, subprocess, sys, threading, time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'.image_runtime_pkgs'))
import paramiko

OUT=Path('output/sdxl-crossdistill/a50k_dataset_20261123')
DEST='/data/shenglin.zeng/a50k_20261123'
HOST='172.18.32.81'
RCLONE=str(Path('tmp/rclone_runtime/rclone-v1.76.0-windows-amd64/rclone.exe').resolve())
raw=Path('C:/Users/sherl/Desktop/login.txt').read_text(encoding='utf-8-sig')
AUTH={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
ENV=dict(os.environ,RCLONE_CONFIG='NUL')
for name,host in [('DEST',HOST),('SOURCE','172.18.33.60')]:
    user,password=AUTH[host]
    for key,val in dict(TYPE='sftp',HOST=host,USER=user,PORT='22',
                        PASS=subprocess.check_output([RCLONE,'obscure','-'],input=password+'\n',text=True).strip()).items():
        ENV[f'RCLONE_CONFIG_{name}_{key}']=val
lock=threading.Lock()
def remote(command,host=HOST):
    c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host,username=AUTH[host][0],password=AUTH[host][1],timeout=20,look_for_keys=False,allow_agent=False)
    c.get_transport().set_keepalive(15)
    _,o,e=c.exec_command(command,timeout=600);output=o.read().decode();err=e.read().decode();rc=o.channel.recv_exit_status();c.close()
    if rc:raise RuntimeError(f'SSH {host} exit {rc}: '+err[:300])
    return output.strip()
def rc(*args):
    subprocess.run([RCLONE,*args,'--log-level','ERROR','--retries','2','--low-level-retries','3','--contimeout','20s','--timeout','90s'],env=ENV,check=True)
def record(item):
    with lock:
        with (OUT/'传输与下载校验记录.txt').open('a',encoding='utf-8') as f:
            f.write(json.dumps(dict(time=time.strftime('%Y-%m-%d %H:%M:%S'),**item),ensure_ascii=False)+'\n')
def upload(path, dest):
    path=Path(path);sha=hashlib.sha256(path.read_bytes()).hexdigest()
    rc('copyto',str(path),'dest:'+dest)
    got=remote('sha256sum '+shlex.quote(dest)).split()[0]
    assert sha==got,(path,got)
    record(dict(source_host='local Windows',source=str(path.resolve()),destination_host=HOST,destination=dest,sha256=sha,verified=True))
def shard(spec):
    name=spec['name'];dest=DEST+'/datasets/sources/'+name
    existing=remote('if [ -f '+shlex.quote(dest)+' ]; then sha256sum '+shlex.quote(dest)+'; fi')
    if existing:
        assert existing.split()[0]==spec['sha256'],name+' invalid existing destination'
        print('ALREADY_VERIFIED',name,flush=True);return
    partial=dest+'.partial'
    if spec['shard'] in (2,3):
        source='/data5/shenglin.zeng/checkpoints/a800_archive_20261106/datasets/20k_sources/'+name
        assert remote('sha256sum '+shlex.quote(source),'172.18.33.60').split()[0]==spec['sha256']
        print('COPY_EXISTING',name,flush=True)
        rc('copyto','source:'+source,'dest:'+partial)
        origin=dict(source_host='172.18.33.60',source=source)
    else:
        print('DOWNLOAD',name,flush=True)
        rc('copyurl',spec['url'],'dest:'+partial)
        origin=dict(source_host='huggingface.co',source=spec['url'])
    got=remote('sha256sum '+shlex.quote(partial)).split()[0]
    size=int(remote('stat -c %s '+shlex.quote(partial)))
    assert got==spec['sha256'] and size==spec['size'],(name,got,size)
    remote('mv -- '+shlex.quote(partial)+' '+shlex.quote(dest))
    record(dict(**origin,destination_host=HOST,destination=dest,sha256=got,bytes=size,verified=True))
    print('VERIFIED',name,size,flush=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--upload',nargs=2,action='append',default=[])
    ap.add_argument('--download',action='store_true');ap.add_argument('--fallback',action='store_true');args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    remote('mkdir -p '+DEST+'/datasets/sources '+DEST+'/code '+DEST+'/records')
    for src,dst in args.upload:upload(src,dst)
    if args.download:
        specs=json.loads((OUT/'sources.json').read_text())['files']
        specs=[s for s in specs if not s['fallback_only'] or args.fallback]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(shard,specs))
        upload(OUT/'sources.json',DEST+'/datasets/sources/sources.json')
        upload(OUT/'传输与下载校验记录.txt',DEST+'/records/transfer_verified.txt')
        print('ALL_SOURCE_SHARDS_VERIFIED',flush=True)
if __name__=='__main__':main()
