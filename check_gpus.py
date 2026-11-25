import sys, re, json, concurrent.futures
sys.path.insert(0, '.runtime_pkgs')
import paramiko
from pathlib import Path

raw = Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds = {ip:(user,pw.strip()) for ip,user,pw in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
command = '''hostname; date -Is; command -v nvidia-smi; nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv; nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv; nvidia-smi -L'''
def connect(ip, sock=None):
    client=paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    user,pw=creds[ip]
    client.connect(ip,username=user,password=pw,timeout=10,auth_timeout=15,banner_timeout=15,look_for_keys=False,allow_agent=False,sock=sock)
    return client
def check(ip):
    clients=[]
    errors=[]
    routes=[None] if ip!='172.18.132.41' else [None,'172.18.33.9','172.18.32.78','172.18.33.60']
    for jump in routes:
        try:
            sock=None
            if jump:
                j=connect(jump); clients.append(j)
                sock=j.get_transport().open_channel('direct-tcpip',(ip,22),('127.0.0.1',0),timeout=10)
            c=connect(ip,sock); clients.append(c)
            _,out,err=c.exec_command(command,timeout=25)
            return {'ip':ip,'route':jump or 'direct','output':out.read().decode(errors='replace'),'stderr':err.read().decode(errors='replace')}
        except Exception as e:
            errors.append({'route':jump or 'direct','error':str(e)})
        finally:
            for c in clients: c.close()
            clients=[]
    return {'ip':ip,'errors':errors}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    results=list(pool.map(check,creds))
Path('gpu_check_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False,indent=2))
