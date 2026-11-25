import csv,json,sys,os
from pathlib import Path
base=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(base/'output/sdxl-crossdistill/.matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
s=base/'output/sdxl-crossdistill/teacher_turning_resample_20261026/results512'
w=base/'output/dymd_prior_8_20261101/analysis'
rows=list(csv.DictReader((s/'curves.csv').open(encoding='utf-8-sig')))
t=np.array([float(r['native_t']) for r in rows]);u=np.array([float(r['u']) for r in rows])
angle=np.array([float(r['angle_rad_mean']) for r in rows]);rate=np.array([float(r['rate_rad_per_u_mean']) for r in rows])
d=json.loads((w/'summary.json').read_text())['metrics']
fig,axs=plt.subplots(1,3,figsize=(16,4.6),layout='constrained')
ax=axs[0];ax.plot(t,angle,color='#007a99');ax.set(xlim=(999,0),xlabel='SDXL native timestep (high noise -> low)',ylabel='Angle per solver interval (rad)',title='SDXL 512: raw interval angle')
ax=axs[1];ax.plot(u,rate,color='#007a99',label='SDXL: full field, CFG 8 (512)')
for key,label,style in [('cfg5/future_centered','PF-Wan: temporal centered, CFG 5 (32)','-'),('cfg5/future_uncentered','PF-Wan: future uncentered, CFG 5 (32)','--')]:
 m=d[key];ax.plot(m['sigma_mid_grid'],m['prefix_mean_curves']['32'],style,label=label,color='#d27a18',lw=1.7)
ax.axvspan(.935957551,1,color='gray',alpha=.12);ax.axvline(.935957551,color='gray',ls=':',lw=1)
ax.text(.935,.97,'SDXL ends here',ha='right',va='top',transform=ax.get_xaxis_transform(),fontsize=9)
ax.set(xlim=(0,1),ylim=(0,None),xlabel='Linear-mixture noise: SDXL u / PF-Wan sigma',ylabel='Angle / noise interval (rad)',title='Same axis; different models and protocols');ax.legend(fontsize=8,loc='upper left')
old=json.loads((base/'output/sdxl-crossdistill/hparam_pilot/teacher_turning/result.json').read_text())
ax=axs[2];ug=np.array(old['u_grid']);mask=ug>=.8
for key,color,label in [('precision_dt10_fp32:rf_full','#8c5398','FP32 subset'),('fine_dt5_bf16:rf_full','#318057','Half-step BF16 subset')]:
 c=old['matched_controls'][key]
 ax.plot(ug[mask],np.array(c['matched_base_curve'])[mask],'--',color=color,label=label+' baseline')
 ax.plot(ug[mask],np.array(c['control_curve'])[mask],color=color,label=label+' control')
ax.set(xlabel='SDXL linear-mixture noise u',ylabel='Turning rate (smoothed; old matched subsets)',title='Old controls: amplitude is sensitive');ax.legend(fontsize=8)
for ax in axs:ax.grid(alpha=.2)
fig.suptitle('Archived SDXL results revisited | no new GPU run | Wan balanced prefix 32 of 36',fontsize=13)
out=base/'output/dymd_prior_8_20261101/analysis/sdxl_wan_comparison.png'
fig.savefig(out,dpi=160);print(out)
print('mean angle / rate first,last:',angle[[0,-1]],rate[[0,-1]])
print('SDXL region ratios',47.43108976024683/11.06757279117416,47.43108976024683/6.409922787000269)
