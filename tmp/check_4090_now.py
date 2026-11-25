import sys, re, json, concurrent.futures
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / '.image_runtime_pkgs'))
import paramiko

raw = Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds = {ip: (u, p.strip()) for ip, u, p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)', raw)}
command = '''hostname
date -Is
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv
nvidia-smi -L
nvidia-smi topo -m
df -h -x tmpfs -x devtmpfs
free -h
command -v python python3 conda
for p in /data*/shenglin.zeng /home/shenglin.zeng; do if test -d "$p"; then ls -ld "$p"; fi; done
'''
def connect(ip, sock=None):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    u, p = creds[ip]
    c.connect(ip, username=u, password=p, timeout=10, auth_timeout=10, banner_timeout=10,
              look_for_keys=False, allow_agent=False, sock=sock)
    return c
def check(ip):
    errors = []
    for route in (None, '172.18.33.60'):
        clients = []
        try:
            sock = None
            if route:
                jump = connect(route); clients.append(jump)
                sock = jump.get_transport().open_channel('direct-tcpip', (ip,22), ('127.0.0.1',0), timeout=10)
            c = connect(ip, sock); clients.append(c)
            _, out, _ = c.exec_command(command, timeout=35)
            out.channel.set_combine_stderr(True)
            return {'ip':ip, 'route':route or 'direct', 'output':out.read().decode(errors='replace')}
        except Exception as e:
            errors.append({'route':route or 'direct','error':str(e)})
        finally:
            for c in reversed(clients): c.close()
    return {'ip':ip,'errors':errors}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    results = list(pool.map(check, ['172.18.132.41', '172.18.32.81']))
Path('output/gpu_4090_check_20261022.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(results,ensure_ascii=False,indent=2))
