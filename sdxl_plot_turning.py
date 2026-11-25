from pathlib import Path
import sys,os,json
os.environ['MPLCONFIGDIR']=str(Path('output/sdxl-crossdistill/.matplotlib').resolve())
sys.path.insert(0,str(Path('.plot_runtime_pkgs').resolve()))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root=Path('output/sdxl-crossdistill/hparam_pilot/teacher_turning');r=json.loads((root/'result.json').read_text())
t=np.array(r['native_t_grid']);mode=r['modes']['base_dt10_bf16'];curves=mode['curves'];primary=curves['rf_full']
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
ax=axes[0,0];ax.fill_between(t,primary['p25'],primary['p75'],color='#3784c7',alpha=.2,label='Across-trajectory IQR')
ax.plot(t,primary['mean'],color='#999',lw=.8,label='Raw mean');ax.plot(t,primary['smoothed_mean'],color='#1464a0',label='Smoothed mean')
ax.axvspan(980,999,color='#d45c36',alpha=.15,label='Largest measured bin: t980-999 (edge)')
ax.set_title('Primary RF-equivalent velocity turning | 32 trajectories');ax.set_ylabel('radians / unit RF-equivalent noise');ax.set_xlim(999,0);ax.legend(fontsize=8)
ax=axes[0,1]
for field,label in [('rf_full','Full image'),('rf_spatial_centered','Spatial mean removed'),('rf_coarse_centered','Coarse structure (16 x 16)')]:
    v=np.array(curves[field]['smoothed_mean']);mask=t>=800;ax.plot(t[mask],v[mask],label=label)
for p in [979,949,919,899]:ax.axvline(p,color='#888',alpha=.25)
ax.set_xlim(999,800);ax.set_title('Early denoising / high-noise region');ax.set_ylabel('radians / unit noise');ax.legend(fontsize=8)
ax=axes[1,0];bins=r['native_t_bins20'];ax.bar([(b['native_t_low']+b['native_t_high'])/2 for b in bins],[b['mean_angle_degrees'] for b in bins],width=18,color='#3784c7')
ax.set_xlim(999,0);ax.set_title('Total direction change in each 20-native-t bin');ax.set_ylabel('mean degrees / bin')
ax=axes[1,1]
for name,label,color in [('fine_dt5_bf16','Step-size check','#297d60'),('precision_dt10_fp32','Precision check','#9b5fa4')]:
    item=r['matched_controls'][name+':rf_full']
    ax.plot(t,item['matched_base_curve'],color=color,ls='--',label=label+' matched base')
    ax.plot(t,item['control_curve'],color=color,label=label+' control')
ax.set_xlim(999,800);ax.set_title('High-noise checks on identical prompt/seed subsets');ax.set_ylabel('radians / unit noise');ax.legend(fontsize=7)
for ax in axes.flat:ax.set_xlabel('SDXL native timestep (denoising left to right)');ax.grid(alpha=.2)
fig.suptitle('SDXL teacher turning profile | DyMD Eq6 adapted to images | CFG 8',fontsize=15)
fig.savefig(root/'turning_profile.png',dpi=150);plt.close(fig)

example=json.loads((root/'early_example.json').read_text());fig,axes=plt.subplots(1,4,figsize=(16,4.7),layout='constrained')
for ax,step in zip(axes,example['native_t']):
    ax.imshow(Image.open(root/f'early_t{step}.png'));ax.axis('off')
    ax.set_title(f"t={step} | RF-equivalent noise={r['coordinate_at_candidates'][str(step)]['u']:.3f}",fontsize=12)
fig.suptitle(f"Earlier candidate clean estimates | seed {example['seed']}\n{example['prompt'][:160]}",fontsize=13)
fig.savefig(root/'earlier_four_stages.png',dpi=130);plt.close(fig)
print('TURNING_FIGURES_SAVED',root.resolve())
