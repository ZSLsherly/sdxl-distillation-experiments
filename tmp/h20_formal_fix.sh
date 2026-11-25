set -eu
. /root/sdxl_research/env.sh
cd /root/autodl-tmp/h20_C_mix1_formal_20261103
python -I - <<'PY'
import os,signal
from pathlib import Path
me=Path.cwd()
def visit(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return
    assert (p/'cwd').resolve()==me and any(b in (p/'cmdline').read_bytes().split(b'\0') for b in (b'queue.py',b'vae_compare.py'))
    os.kill(pid,signal.SIGSTOP)
    for child in (p/'task'/str(pid)/'children').read_text().split():visit(int(child))
    os.kill(pid,signal.SIGKILL)
visit(11397)
PY
mv queue.py run_h20_series.py
mv queue.log startup_import_collision.log
export PYTHONPATH="$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONFAULTHANDLER=1
L=/root/autodl-tmp/h20_gan_mb32/blas126/nvidia/cublas/lib
export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12"
python -c 'import queue; assert "h20_C_mix1_formal" not in queue.__file__; print(queue.__file__)'
nohup python -u run_h20_series.py </dev/null > queue.log 2>&1 & echo QUEUE_PID=$!
