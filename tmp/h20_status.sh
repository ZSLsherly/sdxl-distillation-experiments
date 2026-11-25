set -eu
. /root/sdxl_research/env.sh
cd /root/autodl-tmp/h20_gan_mb32/code
tail -12 h20.log | cut -c1-440
grep 'H20_OOM' h20.log | tail -2 || true
python - <<'PY'
import os,json
from pathlib import Path
p=Path(os.environ['SDXL_ROOT'])/'outputs/h20_gan_mb32_a4'
for n in ('status.json','benchmark.json'):
 if (p/n).exists():print(n,(p/n).read_text()[:2500])
if (p/'rank0.jsonl').exists():
 r=[json.loads(l) for l in (p/'rank0.jsonl').read_text().splitlines() if l.strip()]
 for k in ('student_done','fake','fake_D','gan_backward_probe','heldout_fake_tracking'):
  z=[x for x in r if x['phase']==k];print(k,len(z),[{key:v for key,v in x.items() if key in ('g','fake_update','cycle_seconds','seconds','grad_norm','peak_allocated_GiB','weighted_accumulated_latent_rms')} for x in z[-3:]])
g=Path('gpu.csv')
if g.is_file():
 vals=[int(z.split(',')[1]) for z in g.read_text().splitlines() if len(z.split(','))==3];print('device_max_MiB',max(vals,default=0))
PY
