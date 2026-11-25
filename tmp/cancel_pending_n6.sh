set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import os,signal,json,datetime
root=Path(os.environ['SDXL_ROOT']);job=root/'outputs/A_pcm10_mix1_g160'
pid=422460
cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert cmd.strip()=='python -u run_A10_mix1.py',cmd
train=423755;launcher=423747
assert 'A_pcm10_mix1_g160/model' in Path(f'/proc/{train}/cmdline').read_bytes().decode()
assert 'torch' in Path(f'/proc/{launcher}/cmdline').read_bytes().decode()
code=root/'tmp/A_pcm10_mix1_g160/code/run_A10_mix1.py'
s=code.read_text();needle=" r=evaluate_checkpoint('A_pcm10_mix1',"
assert needle in s
replacement=" if (job/'CANCEL_POST_TRAIN').exists():\n  print('POST_TRAIN_CANCELLED_BY_USER; final weight retained',flush=True)\n  sys.exit(0)\n"+needle
if "if (job/'CANCEL_POST_TRAIN').exists():" not in s:
 code.with_suffix('.py.before_queue_cancel').write_text(s)
 code.write_text(s.replace(needle,replacement,1))
 compile(code.read_text(),str(code),'exec')
(job/'CANCEL_POST_TRAIN').write_text('2026-11-06: User cancelled all pending tasks except current training. No automatic COCO evaluation.\n')
# Terminate only the supervisor, bypassing cleanup that could terminate its independently launched training child.
os.kill(pid,signal.SIGKILL)
for p in (launcher,train):os.kill(p,0)
receipt={'time':datetime.datetime.now().isoformat(),'cancelled_supervisor':pid,'preserved_launcher':launcher,'preserved_training':train,'post_training_evaluation':'cancelled','final_training_save':'unchanged'}
(job/'queue_cancelled.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
print((job/'model/status.json').read_text())
PY
ps -eo pid,ppid,sid,comm,args | awk '$4 ~ /^python|torchrun/ && /run_A10_mix1|A_pcm10_mix1_g160/ {print substr($0,1,220)}'
