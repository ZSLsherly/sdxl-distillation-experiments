set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import os,signal,json,time
root=Path('/root/sdxl_research');job=root/'outputs/A_terminal_low30_g160_20261113'
pid=int((job/'queue.pid').read_text());cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
assert 'run_A_terminal_prior_n13.py' in cmd
launch=1518;worker=1528
assert 'torch.distributed.run' in Path(f'/proc/{launch}/cmdline').read_bytes().decode()
assert 'A_terminal_low30_g160_20261113/model' in Path(f'/proc/{worker}/cmdline').read_bytes().decode()
assert os.getpgid(launch)==launch
(job/'CANCEL_POST_TRAIN').write_text('User cancelled evaluation and replaced training with PCM10, 2026-11-13\n')
state=json.loads((job/'model/status.json').read_text())
os.kill(pid,signal.SIGKILL)
os.killpg(launch,signal.SIGTERM)
for _ in range(20):
 if not Path(f'/proc/{worker}').exists():break
 time.sleep(1)
if Path(f'/proc/{worker}').exists():os.killpg(launch,signal.SIGKILL);time.sleep(2)
assert not Path(f'/proc/{worker}').exists()
receipt={'reason':'User changed PCM weight 1 to 10; restart from original SDXL; no automatic evaluation','last_status':state,'supervisor_stopped':pid,'training_group_stopped':launch}
(job/'REPLACED_BY_PCM10.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
PY
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
