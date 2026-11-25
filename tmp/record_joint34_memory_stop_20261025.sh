set -e
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025
runtime/bin/python - <<'PY'
from pathlib import Path
import json,time
p=Path('outputs/joint34_uniform_10k_2gpu_20261025')
e={'time':time.time(),'reason':'Stopped only this workflow after first student update and resource monitor stalled; parent user.slice memory.high=100GiB, memory.max=112GiB, high events=3211023; host RAM still available. No server limits changed. GPU memory released.','student_updates_completed':0,'fake_updates_completed':5,'server_limits_unchanged':True}
(p/'workflow_stopped.json').write_text(json.dumps(e,indent=2))
with (p/'workflow.jsonl').open('a') as f:f.write(json.dumps({'event':'stopped',**e})+'\n')
print('RECORDED_STOP')
PY
