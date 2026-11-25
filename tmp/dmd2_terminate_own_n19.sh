python3 - <<'PY'
import os,signal
from pathlib import Path
for pid in [455398,455399,455400,455401,455222,455096]:
 try:
  cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
  if b'/data1/shenglin.zeng/dmd2_matched_10k_20261119/' in cmd:
   os.kill(pid,signal.SIGTERM)
 except ProcessLookupError:pass
 except FileNotFoundError:pass
PY
sleep 3
ps -p 455096,455222,455398,455399,455400,455401 -o pid,stat,rss,args
