from pathlib import Path
import json,math,statistics,shutil,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
r=Path('output/sdxl-crossdistill/dmd2_matched_20261119/final_results_20261121')
a_root=Path('output/sdxl-crossdistill/a800_BC_paired_quota_20261102/completed/uniform_control')
ad=next((a_root/'eval').glob('*g000160*'));am=json.loads((ad/'rank0.json').read_text())['samples'];am={x['original_index']:x for x in am}
cases=[]
for p in sorted((r/'eval/g000160').glob('rank*.json')): cases+=json.loads(p.read_text())['samples']
cases.sort(key=lambda x:(x['split']!='train',[7084,5122,8109,592,986,10099,14585,335].index(x['original_index'])))
(r/'A_reference').mkdir(exist_ok=True)
for x in cases:
 ref=am[x['original_index']];assert (x['prompt'],x['seed'],x['split'])==(ref['prompt'],ref['seed'],ref['split'])
 src=ad/ref['image'];dst=r/'A_reference'/ref['image'];shutil.copy2(src,dst)
 x['A_image']=dst.relative_to(r).as_posix();x['A_sha256']=hashlib.sha256(dst.read_bytes()).hexdigest()
 x['paths']=[f'eval/g000000/{x["image"]}',x['A_image'],f'eval/g000160/{x["image"]}']
 for p in x['paths']:assert Image.open(r/p).size==(1024,1024)
(r/'comparison_manifest.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
for split in ['train','holdout']:
 rows=[x for x in cases if x['split']==split]
 fig,axes=plt.subplots(4,3,figsize=(12,16),dpi=110)
 for i,x in enumerate(rows):
  for j,p in enumerate(x['paths']):
   axes[i,j].imshow(Image.open(r/p));axes[i,j].axis('off')
   axes[i,j].set_title((['SDXL G0 / 4 steps','A G160 / F320','DMD2 G160 / F800'][j] if i==0 else '')+'\n'+f'{split} #{x["original_index"]}',fontsize=10)
 fig.tight_layout(pad=.4);fig.savefig(r/f'{split}_comparison.jpg',dpi=110);plt.close(fig)
logs=[json.loads(s) for s in (r/'rank0.jsonl').read_text().splitlines()];gs=[x for x in logs if x['student_updated']]
assert len(logs)==800 and len(gs)==160
for rank in range(4):
 rr=[json.loads(s) for s in (r/f'rank{rank}.jsonl').read_text().splitlines()]
 assert len(rr)==800 and rr[-1]['F']==800 and sum(x['student_updated'] for x in rr)==160
 assert all(math.isfinite(v) for x in rr for v in x.values() if isinstance(v,float))
fig,axs=plt.subplots(2,3,figsize=(15,8),dpi=130)
spec=[(gs,'student_dm','Student DM',False),(gs,'student_gan','Student GAN (raw; weight .005)',False),(logs,'fake_mse','Fake epsilon MSE',False),(logs,'fake_gan','Fake GAN (raw; weight .01)',False),(gs,'student_grad','Student pre-clip gradient norm',True),(logs,'fake_grad','Fake pre-clip gradient norm',True)]
for ax,(data,key,title,logy) in zip(axs.flat,spec):
 xx=[v['G'] if data is gs else (v['iteration']+1)/5 for v in data];yy=np.array([v[key] for v in data]);win=10 if data is gs else 50
 ax.plot(xx,yy,alpha=.22,lw=.7);ax.plot(xx[win-1:],np.convolve(yy,np.ones(win)/win,'valid'),lw=1.7,label=f'{win}-update moving mean')
 if logy:ax.set_yscale('log');ax.axhline(1,color='red',ls='--',lw=.8,label='clip=1')
 ax.set_title(title);ax.set_xlabel('Student update / equivalent cycle');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.tight_layout();fig.savefig(r/'loss_curves.png');plt.close(fig)
summary={'complete':json.loads((r/'complete.json').read_text()),'last_student':gs[-1],'last_fake':logs[-1],'train_seconds':sum(v['seconds'] for v in logs),'peak_GiB':max(v['peak_GiB'] for v in logs),'all_four_ranks_finite':True,'evaluation_images_G160':len(cases)}
for label,data in [('first20G',gs[:20]),('last20G',gs[-20:]),('first100F',logs[:100]),('last100F',logs[-100:])]:
 keys=['student_dm','student_gan','student_grad'] if label.endswith('G') else ['fake_mse','fake_gan','fake_grad']
 summary[label]={k:statistics.mean(v[k] for v in data) for k in keys}
summary['fake_mse_by_depth']={str(d):{label:statistics.mean(x['fake_mse'] for x in logs if lo<=x['G']<=hi and x['depth']==d) for label,lo,hi in [('early',1,20),('late',141,160)]} for d in range(4)}
(r/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('FIGURES_AND_METRICS_READY',r)
