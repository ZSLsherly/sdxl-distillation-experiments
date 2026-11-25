set -eu
. /root/sdxl_research/env.sh
dir="$SDXL_ROOT/outputs/interval_D_20261103/D"
test -f "$dir/config.json"
python - "$dir" <<'PY'
import json,sys
from pathlib import Path
d=Path(sys.argv[1]); c=json.loads((d/'config.json').read_text()); assert 'teacher_interval_quota_D_979' in c['turning_profile']
s=json.loads((d/'status.json').read_text())
if s['status']=='running':
    (d/'STOP_REQUEST').touch()
print('D',s['status'],s['student_updates'],'stop requested',(d/'STOP_REQUEST').is_file())
PY
