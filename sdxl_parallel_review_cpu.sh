set -e
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
mkdir -p outputs/parallel_review_20261017
python - <<'PY'
import tarfile
with tarfile.open('outputs/parallel_review_source.tar.gz') as t:t.extractall('outputs/parallel_review_20261017',filter='data')
PY
cd outputs/parallel_review_20261017
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python test_batching.py > test_result.txt 2>&1
tail -n 5 test_result.txt
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python - <<'PY'
import torch,copy,json,random
from pixel import RGBDiscriminator
from objectives import discriminator_loss
from batching import random_inputs
torch.set_num_threads(2);torch.manual_seed(81)
m=RGBDiscriminator();a=copy.deepcopy(m);b=copy.deepcopy(m)
x=torch.randn(4,3,32,32);y=torch.randn_like(x);text=torch.randn(4,1280)
def calc(d,bs):
 for i in range(0,4,bs):
  pr=d(x[i:i+bs],text[i:i+bs]);pf=d(y[i:i+bs],text[i:i+bs]);(discriminator_loss(pr,pf)*bs/4).backward()
 return torch.cat([p.grad.flatten() for p in d.parameters() if p.grad is not None])
ga=calc(a,1);gb=calc(b,2)
print('SN_TRAIN_GRAD_RELATIVE_L2',float((ga-gb).norm()/ga.norm()))
print('SN_TRAIN_GRAD_COSINE',float(torch.nn.functional.cosine_similarity(ga[None],gb[None])))
# Freeze the same SN buffers: then ordinary batching should recover equivalence.
c=RGBDiscriminator().eval();d=RGBDiscriminator().eval();c.load_state_dict(a.state_dict());d.load_state_dict(a.state_dict())
gc=calc(c,1);gd=calc(d,2);print('SN_FROZEN_GRAD_RELATIVE_L2',float((gc-gd).norm()/gc.norm()))
for phase,offset in [('fake',100000),('student',200000)]:
 r=random_inputs(list(range(4)),phase,'cpu')
 oldc=[(random.Random(i+(0 if phase=='fake' else 100)).randrange(81),random.Random(i+(9 if phase=='fake' else 109)).randrange(81)) for i in range(4)]
 print('AUGMENT',phase,json.dumps({'old_crops':oldc,'new_crops':r['crops'],'old_D_flips':[i%2==0 for i in range(4)],'new_flips':r['flips'].reshape(-1).tolist()}))
PY
