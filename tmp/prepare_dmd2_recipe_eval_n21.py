from pathlib import Path
import shutil,json
base=Path('output/sdxl-crossdistill/dmd2_matched_20261119/coco10k_official_20261121')
root=Path('output/sdxl-crossdistill/dmd2_official_recipe_20261121')
dest=root/'coco10k';dest.mkdir(exist_ok=True)
for folder in ('code','inputs'):
    shutil.copytree(base/folder,dest/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
j=json.loads((base/'job.json').read_text(encoding='utf-8'));cfg=json.loads((root/'config.json').read_text(encoding='utf-8'))
j.update(checkpoint='/data1/shenglin.zeng/dmd2_official_recipe_20261121/runs/formal_g160/student.pt',
         gpu_uuids=cfg['gpu_uuids'],method='DMD2_official_recipe_G160')
(dest/'job.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
p=dest/'code/evaluate_dmd2.py'
s=p.read_text(encoding='utf-8').replace("'DMD2_matched_G160':values",'METHOD:values')
p.write_text(s,encoding='utf-8')
print('COCO10K_CONFIG_READY_NO_EVALUATION_STARTED')
