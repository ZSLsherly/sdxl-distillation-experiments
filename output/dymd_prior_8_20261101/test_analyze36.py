import ast,json,math,subprocess,sys,tempfile
from pathlib import Path
from random import Random
import torch
B=Path(__file__).resolve().parent
plan=json.loads((B/'sample_plan.json').read_text())
assert len(plan)==128
assert len({r['filename'] for r in plan})==128
assert {(r['condition'],r['seed']) for r in plan}=={(c,s) for c in range(8) for s in range(16)}
assert [r['seed'] for r in plan[:8]]==[0]*8
tree=ast.parse((B/'profile_official.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare')
scope={'BY_FILE':{Path(r['filename']).name:r for r in plan},'Path':Path,'Random':Random,'torch':torch,'RANK':1,
       'orig_prepare':lambda self,batch:torch.randn(64,generator=self.generator)}
exec(compile(ast.Module(body=[node],type_ignores=[]),'<prepare>','exec'),scope)
class Fake:device='cpu'
samples=[]
for index in [0,8,120]:
 f=Fake();f.seed=0;f.generator=torch.Generator().manual_seed(0)
 x=scope['prepare'](f,{'filename':plan[index]['filename']})
 assert f.seed==plan[index]['seed'] and f._profile_index==index
 expected=torch.randn(64,generator=torch.Generator().manual_seed(plan[index]['seed']))
 assert torch.equal(x,expected)
 samples.append(x)
assert not torch.equal(samples[0],samples[1])
print('PASS: 128 unique outputs; balanced 8x16 design; actual noise generator uses seeds 0,1,15')
with tempfile.TemporaryDirectory(prefix='dymd128_test_') as temp:
 b=Path(temp);p=b/'profiles';p.mkdir();(b/'sample_plan.json').write_text(json.dumps(plan))
 for item in plan:
  i=item['sample'];rows=[]
  for branch in ['cfg5','conditional','negative']:
   for kind in ['future_centered','future_uncentered','all_uncentered']:
    for k in range(40):
     sigma=1-.9*k/39;r={'sample':i,'step':k,'branch':branch,'centering':kind,'sigma':sigma}
     if k:
      prev=1-.9*(k-1)/39;mid=(sigma+prev)/2
      r.update(sigma_previous=prev,sigma_mid=mid,kappa=(1+10*mid**4)*(1+item['condition']*.01+item['seed']*.001))
     rows.append(r)
  (p/f'trajectory_{i:02d}.jsonl').write_text('\n'.join(map(json.dumps,rows)))
  (p/f'trajectory_{i:02d}.complete').touch()
 result=subprocess.run([sys.executable,str(B/'analyze36.py'),str(b)],capture_output=True,text=True)
 assert result.returncode==0,result.stderr[-3000:]
 summary=json.loads((b/'analysis/summary.json').read_text())
 assert summary['n']==128 and summary['unique_conditions']==8
 m=summary['metrics']['cfg5/future_centered']
 assert len(m['per_sample_peak_sigma'])==128 and len(m['per_condition_mean_peak_sigma'])==8
 assert m['mean_curve_peak_sigma']>.9
 assert (b/'analysis/turning_profiles.png').stat().st_size>1000
 result=subprocess.run([sys.executable,str(B/'analyze36.py'),str(b),'8'],capture_output=True,text=True)
 assert result.returncode==0,result.stderr[-3000:]
 preview=json.loads((b/'analysis_preview_008/summary.json').read_text())
 assert preview['n']==8 and preview['preliminary'] and preview['seeds_per_condition']==1
 assert list(preview['metrics']['cfg5/future_centered']['prefix_mean_curves'])==['8']
 print('PASS: complete 128-trajectory synthetic aggregation, scene grouping, peak detection and figure export')
 result=subprocess.run([sys.executable,str(B/'analyze36.py'),str(b),'36'],capture_output=True,text=True)
 assert result.returncode==0,result.stderr[-3000:]
 final=json.loads((b/'analysis/summary.json').read_text())
 assert final['n']==36 and final['target_n']==36 and not final['preliminary']
 assert final['condition_counts']=={str(c):(5 if c<4 else 4) for c in range(8)}
 import csv
 with (b/'analysis/cfg5_future_centered.csv').open() as f:
  for row in csv.DictReader(f):
   x=float(row['sigma_mid']);expected=(1+10*x**4)*(1+.01*3.5+.001*1.75)
   assert abs(float(row['mean'])-expected)<1e-10
 print('PASS: 36-sample unequal seed counts use exact equal-condition mean')


print('Compatible existing balanced-prefix tests passed')
