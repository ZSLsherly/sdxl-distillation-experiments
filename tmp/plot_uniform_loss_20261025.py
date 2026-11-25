from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'.plot_runtime_pkgs'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=json.loads((root/'output/sdxl-crossdistill/uniform10k_results_20261025/results.json').read_text())
rows=[x for x in r['rows']['0'] if x['phase']=='student_done']
cfg=r['config'];out=root/'output/sdxl-crossdistill/coco_uniform_g160_20261025'
fig,axs=plt.subplots(2,2,figsize=(13,8.4),layout='constrained')
g=np.array([x['g'] for x in rows]);pcm=np.array([x['pcm_loss'] for x in rows])
ax=axs[0,0];ax.plot(g,pcm,color='#2563eb',alpha=.55,lw=1,label='Raw PCM')
ax.plot(g[7:],np.convolve(pcm,np.ones(8)/8,'valid'),color='#1e40af',lw=2,label='8-update mean')
ax.set_ylim(bottom=0);ax.set_title('PCM increases, but remains below 0.001 raw');ax.set_ylabel('Raw PCM loss');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.legend()
sec=ax.secondary_yaxis('right',functions=(lambda x:x*cfg['pcm_weight'],lambda x:x/cfg['pcm_weight']));sec.set_ylabel('Weighted PCM (x10)')
gan=[x for x in rows if x['gan']];ax=axs[0,1]
ax.plot([x['g'] for x in gan],[x['gan_loss'] for x in gan],'.-',color='#c2410c',lw=1,label='Stage 4 active updates only')
ax.set_ylim(bottom=0);ax.set_title('GAN loss does not grow in the late phase');ax.set_ylabel('Raw generator loss: -D(fake)');ax.legend()
w=cfg['gan_weight'];sec=ax.secondary_yaxis('right',functions=(lambda x:x*w,lambda x:x/w));sec.set_ylabel('Weighted GAN (x0.005274)')
colors=['#2563eb','#059669','#c2410c'];ax=axs[1,0]
for s,c in zip((2,3,4),colors):
 a=[x for x in rows if x['stage']==s];ax.plot([x['g'] for x in a],[x['grad_norm'] for x in a],'.-',color=c,lw=1,label=f'Stage {s}')
ax.set_ylim(0,6.6);ax.set_title('Student gradient: separate stage curves');ax.set_ylabel('Global L2 norm before clipping');ax.legend();ax.text(.98,.95,'Clip threshold = 10\n0 / 160 updates clipped',transform=ax.transAxes,ha='right',va='top')
ax=axs[1,1];data=[[x['grad_norm'] for x in rows if x['stage']==s and x['g']>=121] for s in (2,3,4)]
bp=ax.boxplot(data,tick_labels=['Stage 2','Stage 3','Stage 4'],patch_artist=True)
for box,c in zip(bp['boxes'],colors):box.set_facecolor(c);box.set_alpha(.3)
for i,(a,c) in enumerate(zip(data,colors),1):ax.scatter(np.linspace(i-.09,i+.09,len(a)),a,s=18,color=c,alpha=.65)
ax.set_ylim(bottom=0);ax.set_title('Late phase G121-160: stage-conditioned variation');ax.set_ylabel('Global L2 norm before clipping')
for ax in axs.flat:ax.grid(axis='y',alpha=.18)
for ax in (axs[0,0],axs[0,1],axs[1,0]):ax.set_xlabel('Student update G')
fig.suptitle('Uniform 999 / 749 / 499 / 249 | G160 | logged training diagnostics',fontsize=15)
fig.savefig(out/'loss_by_stage.png',dpi=170);fig.savefig(out/'loss_by_stage.pdf');print(out/'loss_by_stage.png')
