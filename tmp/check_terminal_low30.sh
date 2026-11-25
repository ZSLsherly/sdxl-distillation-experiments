set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import os,json
root=Path(os.environ['SDXL_ROOT']);p=root/'tmp/deploy_terminal_low30.sh';p.write_text(p.read_text())
job=root/'outputs/C_mix1_pcm_dm4_fake4_low30_g160'
pid=int((job/'queue.pid').read_text());os.kill(pid,0)
print('QUEUE_PID',pid,Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
print((job/'queue.log').read_text());print((job/'cpu_validation.json').read_text())
a=json.loads((root/'outputs/A_pcm10_mix1_g160/model/status.json').read_text());print('CURRENT_TRAINING',a['status'],a['student_updates'])
cfg=json.loads((job/'effective_config.json').read_text());print('CONFIG',{k:cfg.get(k) for k in ['fake_updates','student_lr','fake_lr','terminal_only_dm','matched_wall_seconds','fake_student_mix','fake_mix_start_g','pcm_weight','max_student_updates','batch','micro_batch','rollout_batch','turning_profile']})
assert not (job/'model').exists()
PY
