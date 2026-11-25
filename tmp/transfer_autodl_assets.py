"""Copy authorized assets at a limited rate; destination credentials via stdin."""
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, '/data5/shenglin.zeng/tools/eval_transfer_20261025')
import paramiko

request = json.loads(sys.stdin.readline())
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('region-9.autodl.pro', port=25619, username=request['user'], password=request.pop('password'), timeout=20, look_for_keys=False, allow_agent=False)
client.get_transport().set_keepalive(20)
dest = '/root/autodl-tmp/sdxl_research/transfers'
root = Path('/data5/shenglin.zeng/sdxl_crossdistill')
files = {
    'environment.tar.gz': root/'exports/turning_20261026/environment.tar.gz',
    'sdxl_sampling.tar': root/'exports/turning_20261026/sdxl_sampling.tar',
    'prepared.pt': root/'datasets/parallel_10k_seed20261018/prepared.pt',
    'dataset_manifest.json': root/'datasets/parallel_10k_seed20261018/manifest.json',
}
results = {}
with client.open_sftp() as sftp:
    for name, source in files.items():
        size = source.stat().st_size
        tmp = dest+'/'+name+'.partial'
        digest = hashlib.sha256(); count = 0; start = time.monotonic(); last = -1
        try:
            complete_size=sftp.stat(dest+'/'+name).st_size
        except FileNotFoundError:
            complete_size=None
        if complete_size is not None:
            if complete_size != size: raise ValueError('Unexpected existing size '+name)
            with source.open('rb') as local:
                for chunk in iter(lambda:local.read(8*1024**2),b''): digest.update(chunk)
            results[name]={'bytes':size,'sha256':digest.hexdigest()}
            print('EXISTING_TO_VERIFY',name,size,flush=True)
            continue
        try: offset=sftp.stat(tmp).st_size
        except FileNotFoundError: offset=0
        if offset>size: raise ValueError('Oversized partial '+name)
        with source.open('rb') as local:
            remaining=offset
            while remaining:
                chunk=local.read(min(8*1024**2,remaining));digest.update(chunk);remaining-=len(chunk)
        count=offset
        print('RESUME',name,'from',offset,flush=True)
        with source.open('rb') as local, sftp.open(tmp, 'ab' if offset else 'wb') as remote:
            remote.set_pipelined(True)
            local.seek(offset)
            while True:
                chunk=local.read(4*1024**2)
                if not chunk: break
                remote.write(chunk); digest.update(chunk); count+=len(chunk)
                wait = (count-offset)/(20*1024**2)-(time.monotonic()-start)
                if wait>0: time.sleep(wait)
                if count//(1024**3)!=last:
                    last=count//(1024**3); print('COPY',name,round(count/1024**3,2),'/',round(size/1024**3,2),'GiB',flush=True)
        if sftp.stat(tmp).st_size != size: raise ValueError('Incomplete transfer '+name)
        sftp.rename(tmp,dest+'/'+name)
        results[name]={'bytes':size,'sha256':digest.hexdigest()}
    with sftp.open(dest+'/manifest.json','w') as f: f.write(json.dumps(results,indent=2))
client.close()
print('TRANSFER_COMPLETE',json.dumps(results),flush=True)
