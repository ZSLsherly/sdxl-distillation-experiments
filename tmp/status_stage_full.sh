set -eu
root=/data1/shenglin.zeng/teacher_turning_20261026/stage_intervention
date -Is
python3 - <<'PY'
import json
from pathlib import Path
p=Path('/data1/shenglin.zeng/teacher_turning_20261026/stage_intervention')
d=json.loads((p/'status.json').read_text())
print('PROGRESS',d['latent_cases'],'/32 prompts',d['pngs'],'/192 PNGs','elapsed',round(d['elapsed']))
for g,v in d['gpus'].items():print('GPU',g,'state',v['state'],'free_GiB',round(v['free']/1024,2),'util',v['util'])
for name in ['complete.json','error.json']:
 if (p/name).exists():print(name,(p/name).read_text())
PY
for f in "$root"/worker*.log;do printf '\n%s\n' "$f";tail -4 "$f";done
