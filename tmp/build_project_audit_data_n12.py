from pathlib import Path
import json,hashlib,html,math
root=Path.cwd();base=root/'output/sdxl-crossdistill';out=base/'project_audit_20261112';out.mkdir(exist_ok=True)
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
rows=[]
def add(key,name,source,obj,hours=None,clip=None,clipn=None,family='boundary'):
 assert all(math.isfinite(obj[k]) for k in ['FID','Patch_FID'])
 rows.append(dict(key=key,name=name,FID=obj['FID'],Patch_FID=obj['Patch_FID'],hours=hours if hours is not None else obj.get('train_wall_hours'),CLIP=clip,CLIP_n=clipn,source=source,family=family,status='G160+COCO10K' if family!='public' else 'public reference'))
s='coco_uniform_g160_20261025/combined_scores.json';d=read(base/s)
for key,name,hrs in [('dmd_only','旧四点：PCM＋DMD',6.07),('gan4','旧四点：末步纯GAN',9.64),('joint4','旧四点：末步GAN＋DMD',5.41),('joint34','旧四点：第3/4步GAN＋DMD',8.05),('joint4_uniform','均匀四点：旧clean目标＋末步GAN',9.41)]:add(key,name,s,d[key],hrs,31.8042755 if key=='joint4_uniform' else None,1024 if key=='joint4_uniform' else None,'legacy')
add('sdxl','原始SDXL 50步',s,d['sdxl'],clip=31.686951,clipn=1024,family='public');add('dmd2','官方DMD2 4步',s,d['dmd2'],clip=31.581163,clipn=1024,family='public')
s='bc_texture_20261102/coco10k_A_complete.json';add('A','A：无先验，PCM×1',s,read(base/s)['scores']['A'],6.8464,31.6747246,1024)
s='bc_texture_20261102/coco10k_BC_complete.json';d=read(base/s)['scores'];add('B','B：高噪声先验',s,d['B'],6.81,31.671,1024);add('C','C：双端先验',s,d['C'],6.82,31.610,1024)
s='dm_pair_coco10k_complete_20261104.json';d=read(base/s)
for key,name,clip,n in [('C_mix1','C＋1%参数混合',31.6357193,1024),('DM234_fake3','C＋1%：PCM＋DM234全选，fake×3',31.925985,128),('DM1_mid4_fake3','C＋1%：首步DM＋随机中段＋末步',31.879019,128)]:add(key,name,s,d[key],clip=clip,clipn=n)
s='all_experiments_20261105/pcm_dm4.json';add('PCM_DM4','C＋1%：PCM＋仅末步学生DM（fake保留中段）',s,read(base/s),5.98225,31.5000973,1024)
s='all_experiments_20261105/a_pcm10.json';add('A_pcm10','A：PCM×10',s,read(base/s)['A_pcm10'],clip=31.4531994,clipn=1024)
s='all_experiments_20261105/dm1_uniform.json';add('DM1_uniform','首步DM＋随机中段＋末步，无先验＋1%',s,read(base/s)['uniform'],clip=31.819804,clipn=128)
add('H20_GAN','C＋1%：额外末步去噪GAN（H20）','实验记录.txt',{'FID':25.667214,'Patch_FID':25.669462},16.779)
s='A_pcm10_patchgan_scores_20261106.json';add('A10_GAN','A×10＋DINO/RGB末步GAN',s,read(base/s)['A_pcm10_patchgan'])
s='all_experiments_20261106/clean_x0_comparison.json';add('A_clean','A：中段改为干净x₀',s,read(base/s)['clean'])
assert len(rows)==19 and sum(r['family']!='public' for r in rows)==17
(out/'experiment_results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
files=['sdxl_research/train.py','sdxl_research/boundary.py','sdxl_research/batching.py','sdxl_research/fake_tracking.py','sdxl_research/training_state.py','sdxl_research/turning_sampling.py','sdxl_research/pixel.py','sdxl_research/sam_pixel.py','sdxl_research/evaluation_core.py','sdxl_research/evaluate.py','tmp/run_terminal_low30.py','tmp/prepare_terminal_low30.py','tmp/prepare_a10_patch_gan.py','tmp/prepare_dm1_ablation.py','tmp/patch_dm234_fake3.py','sdxl_parallel/train.py','sdxl_uniform4_20261022/train.py','output/sdxl-crossdistill/missing_evaluation_20261111/code/run_missing.py','output/sdxl-crossdistill/pcm_dm4_weight_ablation_20261109/train.py','output/sdxl-crossdistill/pcm_dm4_weight_ablation_20261109/terminal_ablation.py']
(out/'source_inventory.json').write_text(json.dumps([dict(path=f,bytes=(root/f).stat().st_size,sha256=hashlib.sha256((root/f).read_bytes()).hexdigest()) for f in files],ensure_ascii=False,indent=2),encoding='utf-8')
print(len(rows),'rows',len(files),'source hashes')
