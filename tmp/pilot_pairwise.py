import json,sys
from pathlib import Path
sys.path.insert(0,'.plot_runtime_pkgs')
import numpy as np
p=Path('output/sdxl-crossdistill/hparam_pilot/assessment/result.json')
if not p.exists():print('WAITING_FOR_RESULT_DOWNLOAD');raise SystemExit
r=json.loads(p.read_text(encoding='utf-8'));rows=r['records'];ref='lr5e7_pcm10';out={}
for name in ['lr25e8_pcm10','lr5e7_pcm1','lr25e8_pcm1']:
 values=[]
 for idx in sorted({x['cache_index'] for x in rows}):
  aa=[x['clip_cosine'] for x in rows if x['method']==ref and x['cache_index']==idx]
  bb=[x['clip_cosine'] for x in rows if x['method']==name and x['cache_index']==idx]
  values.append(float(np.mean(aa)-np.mean(bb)))
 rng=np.random.default_rng(20261017);boot=rng.choice(values,(10000,len(values)),replace=True).mean(1)
 out[name]={'reference_minus_other_clip':float(np.mean(values)),'paired_prompt_bootstrap_95pct_ci':np.quantile(boot,[.025,.975]).tolist()}
print(json.dumps(out,indent=2))
Path('output/sdxl-crossdistill/hparam_pilot/parameter_comparisons.json').write_text(json.dumps(out,indent=2))
