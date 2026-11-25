cd /data5/shenglin.zeng/a50k_20261124
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
/data5/shenglin.zeng/envs/crossdistill/bin/python - <<'PY'
from pathlib import Path
import torch
for run in ('A50K_singlepass','A50K_no_middle_singlepass'):
 for p in sorted((Path('outputs')/run/'training_state').glob('*/complete.json'))[-1:]:
  rows=[]
  for r in range(4):
   state=torch.load(p.parent/f'fake_rank{r}.pt',mmap=True,map_location='cpu',weights_only=True)
   rows.append(state['class_embedding.weight'].clone().float())
  v=torch.cat(rows);print(run,p.parent.name,'shapes',[tuple(x.shape) for x in rows])
  if v.shape[0]==3:print('norm',v.norm(dim=1).tolist(),'row_l2_diff_23_24_34',[float((v[i]-v[j]).norm()) for i,j in [(0,1),(0,2),(1,2)]])
PY
