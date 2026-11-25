from archive_results_n9 import cloud,lab,run,q,N
with cloud() as s:
 script="""from pathlib import Path
from collections import Counter
import re,json
r=Path('/root/autodl-tmp/sdxl_runs/outputs')
cs=Counter(); last={}
for p in r.rglob('*.png'):
 if 'eval' not in p.relative_to(r).parts:continue
 rel=p.relative_to(r)
 par=p.parent
 m=re.search('_g(\\d+)_',par.name);g=int(m[1]) if m else -1
 k=str(par.parent.relative_to(r));cs[k]+=1;last[k]=max(g,last.get(k,-1))
print(cs);print('last',last)
p=next(p for p in r.rglob('rank0.json') if 'eval' in p.parts)
print('META',str(p),str(json.loads(p.read_text()))[:500])
"""
 print(run(s,'/root/sdxl_research/env/bin/python -c '+q(script)))
with lab() as d:
 print(run(d,'ls -lh '+q(N+'/results')+' | tail -20'));print(run(d,'df -Ph '+q(N)))
