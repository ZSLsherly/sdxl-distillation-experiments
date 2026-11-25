set -eu
. /root/sdxl_research/env.sh
n=A_pcm1_terminal_dm_g160_20261113
r="$SDXL_ROOT/outputs/$n"
c="$SDXL_ROOT/tmp/$n/code"
cd "$c"
export PYTHONPATH="$c" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TORCHDYNAMO_DISABLE=1
CUDA_VISIBLE_DEVICES='' python check_A_terminal_only_n13.py
python - <<'PY'
from pathlib import Path
import ast,difflib,json,os,hashlib
r=Path(os.environ['SDXL_ROOT']);job=r/'outputs/A_pcm1_terminal_dm_g160_20261113'
base=r/'outputs/turning_BC_paired_quota_20261102/uniform_control/source'
diff=[]
for name in ('boundary.py','train.py'):
 diff.extend(difflib.unified_diff((base/name).read_text().splitlines(True),Path(name).read_text().splitlines(True),fromfile='A/'+name,tofile='A_terminal/'+name))
(job/'source.diff').write_text(''.join(diff))
for file in Path('.').glob('*.py'):compile(file.read_text(),str(file),'exec')
assert Path('boundary.py').read_text().split('def rollout_boundary_and_final')[1].split('def local_rollout')[0]==(base/'boundary.py').read_text().split('def rollout_boundary_and_final')[1].split('def local_rollout')[0]
train=Path('train.py').read_text();old=(base/'train.py').read_text()
assert train.split('        for fake_repeat in range(repeats):')[1].split('        og.zero_grad(set_to_none=True)')[0]==old.split('        for fake_repeat in range(repeats):')[1].split('        og.zero_grad(set_to_none=True)')[0]
(job/'review.json').write_text(json.dumps(dict(passed=True,baseline_fake_update_block_identical=True,
 baseline_fake_rollout_identical=True,CPU=json.loads((job/'cpu_validation.json').read_text()),
 source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('.').glob('*.py')}),indent=2))
print('SOURCE_REVIEW_PASSED')
PY
test ! -f "$r/queue.pid"
export CUDA_VISIBLE_DEVICES=0
nohup python -u run_A_terminal_only_n13.py >"$r/queue.log" 2>&1 </dev/null &
echo $! >"$r/queue.pid"
sleep 2
date -Iseconds
cat "$r/queue.pid"
tail -5 "$r/queue.log"
kill -0 "$(cat "$r/queue.pid")"
df -h /root/autodl-tmp
