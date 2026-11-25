set -eu
base=/data5/shenglin.zeng/checkpoints/a800_abc_g160_20261102
aux="$base/.abc_g160_transfer_20261102"
python3 - "$aux" <<'PY'
import os,signal,time,sys
from pathlib import Path
w=Path(sys.argv[1])
pid=1643642
def args(p):
    return Path('/proc',str(p),'cmdline').read_bytes().split(b'\0')[:-1]
assert args(pid)==[b'python3',b'-u',os.fsencode(w/'copy.py')]
os.kill(pid,signal.SIGSTOP)
kids=list(map(int,Path('/proc',str(pid),'task',str(pid),'children').read_text().split()))
# The worker threads each spawn their own SSH child.
for t in Path('/proc',str(pid),'task').iterdir():
    kids+=list(map(int,(t/'children').read_text().split()))
kids=set(kids)
for k in kids:
    cmd=args(k)
    assert cmd[0]==b'ssh' and os.fsencode(w/'id') in cmd
for k in kids:os.kill(k,signal.SIGTERM)
os.kill(pid,signal.SIGKILL)
for k in kids:
    for _ in range(40):
        if not Path('/proc',str(k)).exists() or not args(k):break
        time.sleep(.1)
    else:raise RuntimeError('Previous copier not finished')
PY
mv -- "$aux/copy.next.py" "$aux/copy.py"
nohup python3 -u "$aux/copy.py" </dev/null >>"$aux/direct.log" 2>&1 & echo DIRECT_PID=$!
