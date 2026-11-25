set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
import json
from pathlib import Path
root=Path('/root/sdxl_research/code/sdxl_research/configs')
for name in ['boundary_global_noise_g160.json','turning_normalized_u_2gpu_no_offload_g160.json','turning_raw_angle_2gpu_no_offload_g160.json']:
 p=root/name;d=json.loads(p.read_text());d['save_every_updates']=80
 assert d['max_student_updates']==160
 p.write_text(json.dumps(d,indent=2)+'\n');print(name,'save_every_updates',d['save_every_updates'])
PY
