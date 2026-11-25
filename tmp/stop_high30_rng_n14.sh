set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import os,signal,json,time
r=Path(os.environ['SDXL_ROOT']);j=r/'outputs/A_clean_global_high30_g160_20261114'
pid=int((j/'queue.pid').read_text());cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'run_clean_high30_n14.py' in cmd,cmd
children=list(map(int,Path(f'/proc/{pid}/task/{pid}/children').read_text().split()))
assert len(children)==1,children
child=children[0];cc=Path(f'/proc/{child}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert j.name+'/model' in cc and 'torch.distributed.run' in cc,cc
assert os.getpgid(child)==child
status=json.loads((j/'model/status.json').read_text())
os.kill(pid,signal.SIGTERM)
os.killpg(child,signal.SIGTERM)
(j/'SUPERSEDED.json').write_text(json.dumps(dict(reason='Additive stage seed + position aliases across stages; replace prior-time RNG with tuple-keyed seed',
 final_observed_status=status,queue_pid=pid,training_group=child,retain_logs=True,
 excluded_from_comparison=True,replacement='A_clean_global_high30_g160_20261114_v2'),indent=2))
time.sleep(3)
print('STOPPED_OWN_QUEUE',pid,child,'last G',status.get('student_updates'))
print('LOGS_RETAINED',j)
PY
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
