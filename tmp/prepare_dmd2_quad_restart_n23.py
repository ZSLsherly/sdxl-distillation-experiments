from pathlib import Path
import shutil,json
old=Path('output/sdxl-crossdistill/dmd2_official_recipe_20261121')
r=Path('output/sdxl-crossdistill/dmd2_official_recipe_4gpu_20261123');r.mkdir(exist_ok=True)
for folder in ['code','coco10k/code','coco10k/inputs']:
 if not (r/folder).exists():shutil.copytree(old/folder,r/folder,ignore=shutil.ignore_patterns('__pycache__'))
c=json.loads((old/'config.json').read_text(encoding='utf-8'))
c.update(name='DMD2_official_recipe_4gpu_restart_20261123',gpus=[1,2,3,5],
 gpu_uuids=['GPU-dee80cca-d29a-9086-d006-b76ba6ee093e','GPU-7328ece9-841e-484d-9e72-d5b762fdce95','GPU-c374184a-ab5a-681a-ed48-025bed2da39b','GPU-4fba4d7f-66e5-aa77-7210-3803fe7190df'],
 smoke_max_allocated_GiB=39.,smoke_micro_candidates=[8,4,2],
 checkpoint_policy='final inference weights only; rolling resume checkpoint requires requested preference change')
(r/'config.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
p=r/'code/run_guard.py';s=p.read_text(encoding='utf-8')
s=s.replace("(4,2,1)","tuple(cfg['smoke_micro_candidates'])")
s=s.replace("            selected=micro\n",'''            peak=max(json.loads(line)['peak_GiB'] for rank in range(len(cfg['gpu_uuids']))
                     for line in (r/'runs'/name/f'rank{rank}.jsonl').read_text().splitlines())
            if peak>cfg['smoke_max_allocated_GiB']:
                (r/'runs'/name/'student.pt').unlink()
                print('INSUFFICIENT_HEADROOM_REDUCE_MICRO',micro,peak,flush=True)
                continue
            selected=micro
''')
p.write_text(s,encoding='utf-8')
j=json.loads((old/'coco10k/job.json').read_text(encoding='utf-8'))
j.update(checkpoint='/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123/runs/formal_g160/student.pt',gpu_uuids=c['gpu_uuids'],method='DMD2_official_recipe_4gpu_G160')
(r/'coco10k/job.json').write_text(json.dumps(j,indent=2),encoding='utf-8')
src=Path('tmp/deploy_dmd2_recipe_n21.py').read_text(encoding='utf-8-sig').replace('dmd2_official_recipe_20261121','dmd2_official_recipe_4gpu_20261123')
Path('tmp/deploy_dmd2_quad_n23.py').write_text(src,encoding='utf-8')
print(r.resolve())
