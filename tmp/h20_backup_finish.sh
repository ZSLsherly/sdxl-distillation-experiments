set -eu
umask 077
cd /data5/shenglin.zeng/checkpoints/h20_gan_20261104
python3 - <<'PY'
import tarfile,json,os
from pathlib import Path
p=Path.cwd()
with tarfile.open('experiments.tar') as t:
 for k in t:
  path=Path(k.name)
  if path.is_absolute() or '..' in path.parts or not (k.isdir() or k.isfile() or k.islnk() and not Path(k.linkname).is_absolute() and '..' not in Path(k.linkname).parts):
   raise ValueError(k.name)
 t.extractall()
outs=p/'sdxl_runs/outputs'
a=outs/'h20_C_mix1_terminal_GAN_G160';w=next((a/'weights').glob('*/complete.json'))
assert json.loads(w.read_text())['student_updates']==160
target=w.parent/'student_rank0.pt'
if not target.exists():os.link(p/'student_rank0.pt',target)
assert target.stat().st_size==5135506134
c=outs/'h20_C_mix1_terminal_GAN_G160_coco10k'
assert len(list((c/'features/real').glob('*.npz')))>0
m=json.loads((c/'complete.json').read_text())['scores']['C_mix1_terminal_GAN']
assert m['n']==10000
d=outs/'h20_C_mix1_direct_terminal_GAN_G160'
g=[json.loads(l) for l in (d/'rank0.jsonl').read_text().splitlines() if l.endswith('}')]
assert max(v['g'] for v in g if v['phase']=='student_done')==33
assert len(list((next((d/'eval').glob('*g000032_periodic'))).glob('*.png')))==8
(p/'备份完成.txt').write_text('H20 无卡实例实验备份。额外去噪末步 GAN G160 可推理权重 student_rank0.pt。直接 GAN 到 G33，无可续训权重。包含训练日志、源代码、固定图、COCO10K 实际评分及特征。原始模型及训练集未重复备份。\nG160 SHA256 361ad217704fe6b44d01242ea7b91e6ce4c63c90090e40197da99edd0266e068\n实验归档 SHA256 1678224369f6c966138ae793d1eec330025579796a5a48201bc0e4bd4eb6df12\n',encoding='utf8')
print('LAB_BACKUP_COMPLETE',target,m)
PY
