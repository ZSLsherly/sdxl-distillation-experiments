"""Run a supplied experiment command file over the existing SSH connection."""
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / '.image_runtime_pkgs'))
import paramiko
raw = Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
user, password = next((u, p.strip()) for ip, u, p in re.findall(
    r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)', raw)
    if ip == '172.18.33.60')
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('172.18.33.60', username=user, password=password, timeout=15, look_for_keys=False, allow_agent=False)
_, output, errors = client.exec_command(Path(sys.argv[1]).read_text(encoding='utf-8'), timeout=600)
output.channel.set_combine_stderr(True)
for line in output:
    print(line.rstrip(), flush=True)
code = output.channel.recv_exit_status()
client.close()
sys.exit(code)
