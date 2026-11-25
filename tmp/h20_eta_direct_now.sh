set -eu
cd /root/sdxl_research/outputs
/root/sdxl_research/env/bin/python - <<'PY'
from pathlib import Path
import json,statistics
from datetime import datetime
from zoneinfo import ZoneInfo
now=datetime.now(ZoneInfo('Asia/Shanghai'));print(now.isoformat())
for n in ['h20_C_mix1_direct_terminal_GAN_G160','h20_C_mix1_terminal_GAN_G160']:
 p=Path(n);rows=[json.loads(s) for s in (p/'rank0.jsonl').read_text().splitlines() if s.endswith('}')]
 gs=[r for r in rows if r.get('phase')=='student_done']
 dt=lambda r:datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z')
 print(n,len(gs),gs[-1]['g'],dt(gs[-1]),'mean_last10_wall',(dt(gs[-1])-dt(gs[-11])).total_seconds()/10,'mean_cycle',statistics.mean(r['cycle_seconds'] for r in gs[-10:]))
 print((p/'status.json').read_text()[:420])
 q=p.parent/(n+'_coco10k')
 for f in ('manifest.json','status_0.json','complete.json'):
  s=q/f
  if s.exists():print(f,datetime.fromtimestamp(s.stat().st_mtime,now.tzinfo), (' ' if f=='manifest.json' else s.read_text()[:320]))
PY
tail -2 /root/autodl-tmp/h20_direct_after_coco_20261103/queue.log
