set -eu
cd /data5/shenglin.zeng/checkpoints/h20_gan_20261104
python3 - <<'PY'
from pathlib import Path
import hashlib,json
p=Path.cwd(); root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
def digest(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
for f,exp in [('experiments.tar','1678224369f6c966138ae793d1eec330025579796a5a48201bc0e4bd4eb6df12'),('student_rank0.pt','361ad217704fe6b44d01242ea7b91e6ce4c63c90090e40197da99edd0266e068')]:
 print(f,digest(p/f)==exp,flush=True);assert digest(p/f)==exp
outs=p/'sdxl_runs/outputs'
for name in ('h20_C_mix1_terminal_GAN_G160','h20_C_mix1_direct_terminal_GAN_G160'):
 d=outs/name
 m=json.loads((d/'source_sha256.json').read_text())
 assert all(digest(d/'source'/f)==v for f,v in m.items())
 cfg=json.loads((d/'config.json').read_text())
 data=root/'datasets'/Path(cfg['data_dir']).name
 dm=json.loads((d/'data_manifest.json').read_text())
 print(name,'sources',len(m),'data',data,'manifest_equal',json.loads((data/'manifest.json').read_text())==dm,'cache', (data/'prepared.pt').stat().st_size,'keys',list(dm)[:22],flush=True)
 gs=[json.loads(s) for s in (d/'rank0.jsonl').read_text().splitlines() if s.endswith('}')]
 print('lastG',max(x['g'] for x in gs if x['phase']=='student_done'),'eval',len(list(d.glob('eval/*/rank0.json'))))
 if 'direct_' not in name:
  w=next(d.glob('weights/*/student_rank0.pt')); assert w.samefile(p/'student_rank0.pt')
 print('cache sha',digest(data/'prepared.pt'),flush=True) if 'direct_' not in name else None
c=outs/'h20_C_mix1_terminal_GAN_G160_coco10k'
print(json.loads((c/'complete.json').read_text())['scores'])
print('feature blocks',[(x.name,len(list(x.glob('*.npz')))) for x in (c/'features').iterdir()])
print('local teacher',[(root/'models/sdxl-base-1.0'/x).is_dir() for x in ('unet','vae','scheduler')])
PY
