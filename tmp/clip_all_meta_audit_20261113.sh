python3 - <<'PY'
import json,tarfile
from pathlib import Path
b=Path('/data5/shenglin.zeng/checkpoints')
for arm,run in [('DM234_fake3','c_mix1_dm234_fake3_20261103'),('DM1_mid4_fake3','c_mix1_first_dm_fake3_g160'),('DM1_uniform','dm1_mid4_uniform_mix1_g160')]:
 p=b/'a800_archive_20261105/outputs'/run/'model';c=json.loads((p/'config.json').read_text());w=list((p/'weights').glob('*g000160*'))
 print(arm,'knots',c['schedule'],'weights',len(w),'shape',(w[0]/'shapes_rank0.json').exists())
for filename,arm in [('part_004.tar','A_mid_clean_x0'),('part_015.tar','A_pcm10_patchgan')]:
 with tarfile.open(b/'a800_archive_20261106/evidence'/filename) as t:
  for m in t:
   if m.name.endswith('/'+arm+'/manifest.json'):
    v=json.load(t.extractfile(m));r=v['models'][arm]
    print(arm,'knots',r['config']['schedule'],'shards',r['world_size'],r['shards'])
PY
