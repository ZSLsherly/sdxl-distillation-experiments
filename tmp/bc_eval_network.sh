set -eu
python3 - <<'PY'
import socket,time
try:
 s=socket.create_connection(('connect.nma1.seetacloud.com',14892),timeout=6); print('direct SSH',s.recv(128));s.close()
except OSError as e: print(type(e).__name__,e)
PY
