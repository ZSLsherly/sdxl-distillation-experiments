set -euo pipefail
root=/data1/shenglin.zeng/teacher_turning_20261026
cd "$root"
python3 - <<'PY'
import json,math,zipfile
from pathlib import Path
p=Path('results512');files=sorted(p.glob('base_dt10_bf16_*.json'));assert len(files)==512
seen=set();prompts=set()
for f in files:
 d=json.loads(f.read_text());key=(d['prompt_index'],d['seed']);assert key not in seen;seen.add(key);prompts.add(d['prompt'])
 assert len(d['rows'])==100
 for row in d['rows']:
  for field in row['fields'].values():
   assert field['valid'] and all(math.isfinite(v) for v in field.values() if isinstance(v,float))
assert len(prompts)==128
summary={'passed':True,'trajectories':512,'unique_prompts':128,'seeds':sorted({s for i,s in seen}),'intervals_per_trajectory':100}
(p/'validation_final.json').write_text(json.dumps(summary,indent=2))
with zipfile.ZipFile('results512.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in p.iterdir():
  if f.suffix in ['.json','.jsonl','.log']:z.write(str(f),f.name)
print(json.dumps(summary));print('ZIP_BYTES',Path('results512.zip').stat().st_size)
PY
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits
