from pathlib import Path
import json,math,html,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path.cwd();b=r/'output/sdxl-crossdistill';o=b/'project_audit_20261112'
rows=json.loads((o/'experiment_results.json').read_text());d={x['key']:x for x in rows}
fig,ax=plt.subplots(figsize=(10,6))
labels={'sdxl':'SDXL 50','dmd2':'DMD2 4','joint4_uniform':'Old clean + GAN','A':'A','B':'B','C':'C','C_mix1':'C + 1%','PCM_DM4':'PCM + terminal DM','A_pcm10':'A PCM x10','A10_GAN':'A x10 + GAN','A_clean':'A middle clean','DM234_fake3':'DM234 + fake3','H20_GAN':'C + extra GAN'}
for i,(k,label) in enumerate(labels.items()):
 v=d[k];ax.scatter(v['FID'],v['Patch_FID'],s=42,c='#176b8a' if v['family']!='public' else '#818891')
 offset={'A':(7,10),'C':(7,8),'B':(7,7),'C_mix1':(7,-17),'DM234_fake3':(7,-18),'A10_GAN':(-83,4),'sdxl':(7,10),'dmd2':(7,-15)}.get(k,(7,4))
 ax.annotate(label,(v['FID'],v['Patch_FID']),textcoords='offset points',xytext=offset,fontsize=8)
ax.set(xlabel='COCO10K FID (lower is better)',ylabel='Center-299 Patch FID (lower is better)',xlim=(17,31),ylim=(17,52))
ax.grid(alpha=.18);ax.set_title('Completed historical evaluations — single training seed')
fig.tight_layout();fig.savefig(o/'quality_tradeoffs.png',dpi=170);plt.close(fig)
prod=1.;ws=[]
for t in range(1000):
 beta=(math.sqrt(.00085)+(math.sqrt(.012)-math.sqrt(.00085))*t/999)**2;prod*=1-beta;ws.append(min(1,10*(1-prod)))
fig,ax=plt.subplots(figsize=(9,3.7));ax.plot(range(181),ws[:181],color='#b35a24',linewidth=2);ax.axvspan(20,100,alpha=.12,color='#b35a24');ax.axhline(1,color='gray',linestyle='--',linewidth=.8);ax.set(xlabel='SDXL noise timestep t',ylabel='Multiplier on terminal fake epsilon MSE',ylim=(0,1.12),title='fake_weight_max = 10: terminal low-noise weighting');ax.grid(alpha=.15);fig.tight_layout();fig.savefig(o/'fake_low_noise_weight.png',dpi=170);plt.close(fig)
print('PLOTS_READY')
