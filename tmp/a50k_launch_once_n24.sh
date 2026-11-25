set -eu
cd /data5/shenglin.zeng/a50k_20261124
export TMPDIR="$PWD/cache/TMPDIR" OMP_NUM_THREADS=4
py=/data5/shenglin.zeng/envs/crossdistill/bin/python
"$py" - <<'PY'
from pathlib import Path
from types import SimpleNamespace
from textwrap import dedent
import sys
sys.path.insert(0,'code')
from batching import BatchPlan
s=Path('code/train.py').read_text()
args=SimpleNamespace(batch=64,fake_updates=2)
order=range(50000)
exec(dedent(s[s.index('    milestones = {'):s.index('    mesh = init_device_mesh')]))
students=[];fake_seeds=set()
for i,(start,width) in enumerate(batches):
 plan=BatchPlan(width,4,min(16,width//4))
 positions=[start+j for rank in range(4) for j in plan.rank_positions(0,rank)]
 students.extend(positions)
 for repeat in (0,1):
  fake_seeds.update(pos+50000*repeat for pos in positions)
assert sorted(students)==list(order) and len(fake_seeds)==100000 and len(batches)==784
assert [(i,v) for i,v in enumerate(consumed) if v in milestones]==[(313,20000),(470,30000),(627,40000),(784,50000)]
print('PASS_ONE_PASS',len(students),len(fake_seeds))
PY
test ! -e outputs/A50K_singlepass
nohup "$py" -u code/queue50.py >logs/singlepass_queue.log 2>&1 </dev/null &
echo "singlepass queue PID=$!"
