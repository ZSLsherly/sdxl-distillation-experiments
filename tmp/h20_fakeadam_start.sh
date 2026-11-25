set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_gan_mb32
c="$d/code_rows_fakeadam"
mkdir -p "$c";cp "$d/code_rows"/*.py "$d/code_rows"/*.json "$c/"
cd "$c"
python - <<'PY'
from pathlib import Path
p=Path('train.py');s=p.read_text()
target='                of.step()'
assert s.count(target)==1
s=s.replace(target,"""                if fake_repeat==0 and gcount>0:
                    move_fake_moments(device)
"""+target)
target='        og.zero_grad(set_to_none=True)\n        tick = time.monotonic()'
assert s.count(target)==1
s=s.replace(target,'        move_fake_moments("cpu")\n'+target)
target='    def predict(model, x, t, condition, negative=False, score_stage=None):'
assert s.count(target)==1
s=s.replace(target,"""    def move_fake_moments(destination):
        # Keep FP32 moments intact. No optimizer call occurs while they are off device.
        for state in of.state.values():
            for key in ('exp_avg','exp_avg_sq'):
                if key in state:
                    state[key]=state[key].to(destination)

"""+target)
p.write_text(s)
PY
L="$d/blas126/nvidia/cublas/lib";export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12" PYTHONFAULTHANDLER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
sed -e 's/h20_gan_mb32_a4/h20_gan_mb32_rows_fakeadam/g' run_h20.py > run_offload.py
python -m py_compile train.py
nohup python -u run_offload.py </dev/null > h20.log 2>&1 & echo $!
