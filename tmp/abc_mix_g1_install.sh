set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/tmp/bc_coco_winner_20261102/code"
export PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=''
python - <<'PY'
import os,subprocess,sys,json,time
from pathlib import Path
from launch import flags
from train import arguments
from run_bc_coco_then_winner import job,evalout,work
base=Path.cwd().resolve()
legacy=88668
p=Path('/proc')/str(legacy)
assert p.joinpath('cwd').resolve()==base and p.joinpath('cmdline').read_bytes().split(b'\0')[:-1]==[b'python',b'-u',b'run_bc_coco_then_winner.py']
ticks=p.joinpath('stat').read_text().split()[21]
assert not (job/'late_mix').exists() and not (evalout/'complete.json').exists()
q=Path('run_bc_coco_then_winner.py')
fresh=Path('run_bc_coco_then_winner.next').read_text(encoding='utf-8-sig')
old=q.read_text()
assert 'fake_mix_start_g=121' in old and 'from control_after_bc import run_control' in old
assert 'fake_mix_start_g=1)' in fresh and "assert r['student_mix']==.01" in fresh
t=Path('train.py');tr=t.read_text()
oldprobe='(args.fake_mix_start_g > 1 and gcount == args.fake_mix_start_g)'
newprobe='(gcount == args.fake_mix_start_g or (args.fake_student_mix and gcount == 121))'
assert tr.count(oldprobe)==1
tr=tr.replace(oldprobe,newprobe)
ep=Path('evaluate.py');es=ep.read_text();needle='        if core.OUT == evalout.resolve():\n            run_control()';assert es.count(needle)==1
# The original in-memory parent is blocked inside this metrics subprocess.
# Yield its lock only when all three sets of results are durable.
es=es.replace(needle,needle+f"""
            import os, signal
            parent = Path('/proc/{legacy}')
            if os.getppid() == {legacy} and parent.joinpath('stat').read_text().split()[21] == {ticks!r}:
                from run_bc_coco_then_winner import job
                assert (job/'coco10k_A/complete.json').exists() and (core.OUT/'complete.json').exists()
                waiter = int((job/'g1_waiter.pid').read_text())
                assert b'--wait-lock' in Path(f'/proc/{{waiter}}/cmdline').read_bytes().split(b'\\0')
                print('B/C/A complete; handing the queue lock to the G1 configuration',flush=True)
                os.kill({legacy},signal.SIGTERM)""")
for file,content in [(q,fresh),(t,tr),(ep,es)]:compile(content,str(file),'exec')
def write(file,content):
    stage=file.with_suffix('.new');stage.write_text(content);stage.replace(file)
for arm in ['B','C']:
    cfg=json.loads((Path(os.environ['SDXL_ROOT'])/'code/sdxl_research/configs'/f'turning_{arm}_quota_a800_g160.json').read_text())
    cfg.update(fake_track_pilot=True,fake_student_mix=.01,fake_mix_start_g=1)
    sys.argv=['train','--run','validate',*flags(cfg)]
    args=arguments();assert args.fake_mix_start_g==1 and args.fake_student_mix==.01 and args.max_student_updates==160
    assert args.pair_fake_stages and args.batch==64
    print('VALIDATED flags',arm,args.fake_mix_start_g,args.fake_student_mix)
write(q,fresh);write(t,tr)
with (work/'logs/bc_coco_winner_queue.log').open('a') as log:
    waiter=subprocess.Popen([sys.executable,'-u','run_bc_coco_then_winner.py','--wait-lock'],start_new_session=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
(job/'g1_waiter.pid').write_text(str(waiter.pid))
time.sleep(3)
assert waiter.poll() is None
write(ep,es)
print('ARMED',waiter.pid,'old evaluator remains supervised by',legacy,'until B/C/A finish')
PY
python -m unittest test_fake_tracking
ps -o pid,ppid,stat,cmd -p 88668,103394,"$(cat "$SDXL_ROOT/outputs/bc_winner_late_mix_20261102/g1_waiter.pid")"
