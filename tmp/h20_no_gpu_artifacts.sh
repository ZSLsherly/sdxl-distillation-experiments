set -eu
date -Is
ls -ld /root/autodl-tmp/sdxl_runs /root/sdxl_research/outputs /root/autodl-tmp/h20_direct_after_coco_20261103 || true
df -h /root/autodl-tmp
P=/root/sdxl_research/env/bin/python
test -x "$P" || P=/root/miniconda3/bin/python
"$P" - <<'PY'
import json,os
from pathlib import Path
root=Path('/root/autodl-tmp/sdxl_runs')
for b in (Path('/root/sdxl_research/outputs'),root/'outputs'):
 print('OUTPUT_ROOT',b,b.resolve(),b.exists(),flush=True)
 if b.is_dir():print('MATCHES',[x.name for x in b.glob('h20_C_mix1*')])
b=Path('/root/sdxl_research/outputs')
if not b.is_dir():b=root/'outputs'
for n in ('h20_C_mix1_direct_terminal_GAN_G160','h20_C_mix1_terminal_GAN_G160','h20_C_mix1_terminal_GAN_G160_coco10k','h20_C_mix1_direct_terminal_GAN_G160_coco10k'):
 p=b/n
 print('RUN',n,p.exists())
 if not p.is_dir():continue
 for f in ('status.json','complete.json'):
  q=p/f
  if q.exists():print(f,q.read_text()[:900])
 l=p/'rank0.jsonl'
 if l.exists():
  rows=[json.loads(x) for x in l.read_text().splitlines() if x.endswith('}')]
  for phase in ('student_done','fake','fake_D'):
   rr=[v for v in rows if v.get('phase')==phase]; print(phase,len(rr),{k:v for k,v in rr[-1].items() if k in ('g','fake_update','D_update','timestamp')} if rr else None)
  print('LAST_EVENT',{k:v for k,v in rows[-1].items() if k in ('g','phase','fake_update','timestamp','micro')})
 ev=sorted((p/'eval').glob('*'))
 if ev:
  ok=[d for d in ev if (d/'rank0.json').exists() and len(list(d.glob('*.png')))==8]
  print('EVAL',len(ok),'last',ok[-1].name if ok else None,'refinement milestones',[d.name for d in ok if len(list((d/'refinement').glob('*.png')))==24])
 for q in sorted(p.rglob('*')):
  if q.is_file() and (q.suffix in ('.pt','.safetensors','.ckpt','.pth') or q.name in ('cpu_validation.json','complete.json') and 'weights' in q.parts):
   print('FILE',q.relative_to(p),q.stat().st_size, q.read_text()[:380] if q.suffix=='.json' else '')
print('queue_procs')
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().split(b'\0')
  if any(x.endswith((b'after_coco.py',b'run_references.py',b'run.py',b'train.py',b'after_g160.py')) for x in args): print(p.parent.name,args[:2])
 except OSError:pass
PY
