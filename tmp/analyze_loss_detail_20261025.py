from pathlib import Path
import sys,json,statistics,math
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'.plot_runtime_pkgs'))
import numpy as np
r=json.loads((root/'output/sdxl-crossdistill/uniform10k_results_20261025/results.json').read_text())
gs=[x for x in r['rows']['0'] if x['phase']=='student_done'];cfg=r['config']
def stats(x):
 a=np.asarray(x,float)
 return {'n':len(x),'mean':float(a.mean()),'median':float(np.median(a)),
         'min':float(a.min()),'max':float(a.max()),'p95':float(np.percentile(a,95)),
         'CV':float(a.std()/a.mean()) if a.mean()!=0 else None}
result={'weights':{'pcm':cfg['pcm_weight'],'gan':cfg['gan_weight']},'windows':{},
        'all_student_gradients':stats([x['grad_norm'] for x in gs]),
        'student_updates_above_clip10':sum(x['grad_norm']>10 for x in gs)}
for name,lo,hi in [('early',1,40),('middle',61,100),('late',121,160)]:
 rows=[x for x in gs if lo<=x['g']<=hi];gan=[x for x in rows if x['gan']]
 result['windows'][name]={'raw_PCM':stats([x['pcm_loss'] for x in rows]),
     'weighted_PCM':stats([cfg['pcm_weight']*x['pcm_loss'] for x in rows]),
     'raw_GAN_active':stats([x['gan_loss'] for x in gan]),
     'weighted_GAN_active':stats([cfg['gan_weight']*x['gan_loss'] for x in gan]),
     'gradient_by_stage':{str(stage):stats([x['grad_norm'] for x in rows if x['stage']==stage]) for stage in (2,3,4)}}
late=[x for x in gs if x['g']>=40]
values=np.log([x['grad_norm'] for x in late]);mu=float(values.mean())
between=sum(sum(x['stage']==s for x in late)*(float(np.log([x['grad_norm'] for x in late if x['stage']==s]).mean())-mu)**2 for s in (2,3,4))
result['stage_explained_log_gradient_variance_G40_160']=float(between/((values-mu)**2).sum())
ds=[x for x in r['rows']['0'] if x['phase']=='fake_D']
result['discriminator_windows']={}
for name,lo,hi in [('early',1,200),('late',601,800)]:
 rows=[x for x in ds if lo<=x['D_update']<=hi]
 result['discriminator_windows'][name]={k:stats([x[k] for x in rows if k!='R1_raw' or x['R1_active']]) for k in ['D_hinge','D_real','D_fake','D_dino_real','D_dino_fake','D_rgb_real','D_rgb_fake','D_grad_norm','R1_raw']}
out=root/'output/sdxl-crossdistill/coco_uniform_g160_20261025'
out.mkdir(parents=True,exist_ok=True)
(out/'loss_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
