"""Temporary loopback-only SSH forwarding to this machine's configured proxy."""
import sys,re,socket,select,threading,time,urllib.request,urllib.parse
from pathlib import Path
sys.path.insert(0,'.image_runtime_pkgs')
import paramiko
raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
creds={ip:(u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw)}
proxy=urllib.parse.urlsplit(urllib.request.getproxies().get('https','http://127.0.0.1:7897'))
assert proxy.hostname in ('127.0.0.1','localhost')
c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());u,p=creds['172.18.33.60']
c.connect('172.18.33.60',username=u,password=p,timeout=15,look_for_keys=False,allow_agent=False)
transport=c.get_transport();transport.set_keepalive(20)
def forward(channel):
 sock=None
 try:
  sock=socket.create_connection((proxy.hostname,proxy.port),timeout=20)
  while True:
   ready,_,_=select.select([sock,channel],[],[],30)
   for source in ready:
    buf=source.recv(256*1024)
    if not buf:return
    (channel if source is sock else sock).sendall(buf)
 except Exception as e:print('forward closed',type(e).__name__,flush=True)
 finally:
  channel.close()
  if sock:sock.close()
def handler(channel,origin,server):threading.Thread(target=forward,args=(channel,),daemon=True).start()
transport.request_port_forward('127.0.0.1',18797,handler=handler)
print('REMOTE_LOOPBACK_PROXY_READY 18797',flush=True)
try:
 while transport.is_active() and not Path('sdxl_proxy.stop').exists():time.sleep(1)
finally:c.close()
