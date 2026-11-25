set -eu
r=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
cd "$r/outputs/semantic_references_1024"
export PYTHONPATH="$r/code" LD_LIBRARY_PATH="$r/runtime/lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$r/runtime/bin/python" - <<'PY'
import subprocess
from pathlib import Path
def uid(i):return subprocess.check_output(['nvidia-smi','-i',str(i),'--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
p=Path('run_references.py');s=p.read_text()
needle='if u.strip() != '+repr(uid(4))+' and '
assert needle in s
s=s.replace(needle,'if u.strip() in '+repr((uid(3),uid(6)))+' and ').replace('][:4]','][:1]')
p.write_text(s);print('Eligible only hardware 3/6; one new worker at a time',flush=True)
p=Path('generate.py');s=p.read_text();line="uuid=os.environ['CUDA_VISIBLE_DEVICES']"
s=s.replace(line,line+"\nif uuid not in "+repr((uid(3),uid(6)))+":raise SystemExit('outside designated GPUs 3 and 6')")
p.write_text(s)
PY
cat > resume_after_workers.sh <<'SH'
set -eu
../../runtime/bin/python - <<'PY'
from pathlib import Path
import time,os
while True:
    active=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            if p.stat().st_uid==os.getuid() and (p/'cwd').resolve()==Path.cwd() and b'generate.py' in (p/'cmdline').read_bytes().split(b'\0'):
                active.append(p.name)
        except OSError:pass
    if not active:break
    time.sleep(15)
PY
exec ../../env/bin/python -u run_references.py
SH
nohup bash resume_after_workers.sh >resume.log 2>&1 </dev/null & echo WAIT_ONLY_PID=$!
sleep 1
cat run.log|head -2
test ! -e xfer/transfer.log || tail -2 xfer/transfer.log
ls -l assets.ready 2>/dev/null || true
nvidia-smi -i 4 --query-compute-apps=pid,used_memory --format=csv,noheader
