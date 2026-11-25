set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
python - <<'PY'
from pathlib import Path
import json,os,torch,ast,hashlib
root=Path(os.environ['SDXL_ROOT'])
manifest_dir=root/'datasets/parallel_10k_seed20261018'
manifest=json.loads((manifest_dir/'manifest.json').read_text())
data=torch.load(manifest_dir/'prepared.pt',weights_only=True,mmap=True,map_location='cpu')
assert len(manifest['eval_original_indices'])==4
old="ids=manifest['eval_original_indices'][:8]"
new="ids=manifest['holdout_original_indices'][:8]"
expected={ 'bc_coco_winner_20261102':'1e2f07dd98459cea2dc45fb6e40b748f762c2906924098f534ad06220851dd06',
           'flash_gan_g160_20261102':'8a171e58ec17ed0ca9fd941347ae76c4c661f344042b6364a37ac7ef861c7356'}
def probe_selection(source):
    code=ast.parse(source)
    suite=next(n.body for n in ast.walk(code) if isinstance(n,ast.With) and any(isinstance(st,ast.Assign) and isinstance(st.targets[0],ast.Name) and st.targets[0].id=='ids' for st in n.body))
    assert isinstance(suite[2],ast.Assert)
    ns={'manifest':manifest,'data':data}
    exec(compile(ast.Module(body=suite[:3],type_ignores=[]),'<deployed tracking selector>','exec'),ns)
    return ns['ids'],ns['idx']
paths={}
for sub,h in expected.items():
    path=root/'tmp'/sub/'code/train.py'; raw=path.read_bytes();s=raw.decode()
    assert hashlib.sha256(raw).hexdigest()==h and s.count(old)==1
    try:probe_selection(s)
    except AssertionError:pass
    else:raise RuntimeError('old G1 issue did not reproduce')
    fixed=s.replace(old,new)
    ids,idx=probe_selection(fixed)
    assert len(set(ids))==8 and set(ids).isdisjoint(manifest['train_original_indices'])
    assert all(data['original_indices'][i]==j for i,j in zip(idx,ids))
    compile(fixed,str(path),'exec')
    paths[path]=fixed
    print(sub,'G1 selector before: AssertionError; after: OK',ids,idx)
assert not (root/'outputs/bc_winner_late_mix_20261102/late_mix').exists()
assert not list((root/'outputs/flash_gan_g160_20261102').glob('gan_micro*'))
for path,fixed in paths.items():
    tmp=path.with_suffix('.audit-next');tmp.write_text(fixed);tmp.replace(path)
    assert probe_selection(path.read_text())==(ids,idx)
    print('installed sha256',path,hashlib.sha256(path.read_bytes()).hexdigest())
PY
