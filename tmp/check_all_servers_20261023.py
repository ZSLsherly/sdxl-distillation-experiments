import sys, re, json, concurrent.futures
from pathlib import Path
from datetime import datetime, timezone, timedelta
sys.path.insert(0, str(Path.cwd() / '.image_runtime_pkgs'))
import paramiko

raw = Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds = {ip: (u, p.strip()) for ip, u, p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)', raw)}
command = '''hostname
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,pstate --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
nvidia-smi -L
'''
def connect(ip, sock=None):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    u, p = creds[ip]
    c.connect(ip, username=u, password=p, timeout=8, auth_timeout=8, banner_timeout=8,
              look_for_keys=False, allow_agent=False, sock=sock)
    return c
def check(ip):
    errors = []
    routes = [None] + (['172.18.33.60'] if ip != '172.18.33.60' else [])
    for route in routes:
        clients = []
        try:
            sock = None
            if route:
                jump = connect(route); clients.append(jump)
                sock = jump.get_transport().open_channel('direct-tcpip', (ip,22), ('127.0.0.1',0), timeout=8)
            c = connect(ip, sock); clients.append(c)
            _, out, _ = c.exec_command(command, timeout=25)
            out.channel.set_combine_stderr(True)
            return {'ip':ip, 'route':route or 'direct', 'output':out.read().decode(errors='replace')}
        except Exception as e:
            errors.append({'route':route or 'direct','error':str(e)})
        finally:
            for c in reversed(clients): c.close()
    return {'ip':ip,'errors':errors}
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
    results = list(pool.map(check, creds))
report = {'checked_at':datetime.now(timezone(timedelta(hours=8))).isoformat(), 'servers':results}
Path('output/gpu_all_servers_20261023.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))

