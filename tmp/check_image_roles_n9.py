from archive_results_n9 import cloud,lab,run,q,root,N
from pathlib import Path
import json
rel='outputs/A_pcm10_mix1_g160/model/weights/20261106T190544+0800_g000160_end/student_rank0.pt';dst=N+'/'+rel
with cloud() as s:
 cmd="""from pathlib import Path
from collections import Counter
r=Path('/root/autodl-tmp/sdxl_runs/outputs');c=Counter();e={}
for p in r.rglob('*.png'):
 parts=p.relative_to(r).parts
 kind='训练过程固定出图' if 'eval' in parts else '其他/'+'/'.join(parts[1:3]);c[kind]+=1;e.setdefault(kind,str(p.relative_to(r)))
for k,n in c.most_common(14): print(n,k,e[k])
"""
 print(run(s,'/root/sdxl_research/env/bin/python -c '+q(cmd)),flush=True)
 print('source:',run(s,'stat -c %s '+q(root+rel)+' 2>/dev/null || true'),flush=True)
with lab() as d:
 print('target:',run(d,'stat -c "%s %n" '+q(dst)+' '+q(dst+'.partial')+' 2>/dev/null || true'),flush=True)
 print('open:',run(d,'fuser '+q(dst+'.partial')+' 2>/dev/null || true'),flush=True)
