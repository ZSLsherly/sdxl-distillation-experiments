import ast,json
from pathlib import Path
import torch
import torch.nn.functional as F
scope={'torch':torch,'F':F}
for name,functions in [('sdxl_turning_resample.py',{'observations','turn_metrics'}),('sdxl_turning_batch.py',{'metrics'})]:
 tree=ast.parse(Path(name).read_text())
 selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in functions]
 exec(compile(ast.Module(body=selected,type_ignores=[]),name,'exec'),scope)
scope['FIELDS']=['rf_full','rf_conditional_on_guided_path','rf_spatial_centered','rf_coarse_centered']
keys=['angle_rad','rate_rad_per_u','path_rms_distance','rate_rad_per_path_rms','velocity_rms']
torch.manual_seed(101)
old=scope['observations'](*(torch.randn(4,4,16,16) for _ in range(3)))
new=scope['observations'](*(torch.randn(4,4,16,16) for _ in range(3)))
batched=scope['metrics'](old,new,.02)
errors=[]
for i in range(4):
 one=scope['turn_metrics']({k:(v[i:i+1],s[i:i+1]) for k,(v,s) in old.items()},
                         {k:(v[i:i+1],s[i:i+1]) for k,(v,s) in new.items()},.02)
 for f,name in enumerate(scope['FIELDS']):
  for m,key in enumerate(keys):errors.append(abs(batched[i,f,m].item()-one[name][key]))
assert max(errors)<1e-10,max(errors)
out=Path('output/sdxl-crossdistill/teacher_turning_resample_20261026/batch_metric_check.json')
out.write_text(json.dumps({'passed':True,'compared_values':len(errors),'max_absolute_error':max(errors)},indent=2))
print(out.read_text())
