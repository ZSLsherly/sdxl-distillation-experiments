"""Summarize the fixed crossed condition/seed design without selecting favorable samples."""
import csv,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

b=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
n=int(sys.argv[2]) if len(sys.argv)>2 else 128
assert n in (8,32,36,64,128),n
p=b/'profiles';out=b/('analysis' if n in (36,128) else f'analysis_preview_{n:03d}');out.mkdir(exist_ok=True)
full_plan=json.loads((b/'sample_plan.json').read_text());assert len(full_plan)==128
plan=full_plan[:n]
files=[p/f'trajectory_{i:02d}.jsonl' for i in range(n)]
data=[]
for f in files:
 assert f.with_suffix('.complete').exists(),f
 rows=[json.loads(x) for x in f.read_text().splitlines()]
 assert len(rows)==40*9,(f,len(rows))
 data.extend(rows)
condition_counts={c:sum(r['condition']==c for r in plan) for c in range(8)}
balanced=len(set(condition_counts.values()))==1
seed_label=str(n//8) if balanced else f'{min(condition_counts.values())}-{max(condition_counts.values())}'
summary={'n':n,'target_n':36 if n<=36 else 128,'preliminary':n not in (36,128),'unique_conditions':8,'condition_counts':condition_counts,'mean_weighting':'equal_condition','seeds_per_condition':n//8 if balanced else None,'paper_n':32,'high_noise_threshold':0.9,'metrics':{},'limitations':[
 'Official example set differs from the unspecified 32 paper trajectories.',
 'Smoothing bandwidth and epsilon not disclosed; main plot is unsmoothed.',
 'No velocity evaluated at sigma=0; full-range masses below the final evaluated sigma require extrapolation.',
 'Trajectories reuse eight scenes; they are not independent scene conditions. The final 36 have 4-5 seeds per scene, so the main mean equally weights scenes.',
 'A reproduction discrepancy alone does not establish fabrication.'
]}
fig,axs=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
keys=[('cfg5','future_centered'),('conditional','future_centered'),('cfg5','future_uncentered'),('cfg5','all_uncentered')]
for ax,(branch,kind) in zip(axs.flat,keys):
 matrix=[]
 for i in range(n):
  rows=[r for r in data if r['sample']==i and r['branch']==branch and r['centering']==kind and 'kappa' in r]
  assert len(rows)==39
  rows.sort(key=lambda r:r['sigma_mid'])
  x=np.array([r['sigma_mid'] for r in rows]);y=np.array([r['kappa'] for r in rows])
  matrix.append(y);ax.plot(x,y,alpha=.06,lw=.6)
 matrix=np.array(matrix);mean=matrix.mean(axis=0);q1,q3=np.quantile(matrix,[.25,.75],axis=0)
 condition_curves=np.array([matrix[[r['sample'] for r in plan if r['condition']==c]].mean(axis=0) for c in range(8)])
 mean=condition_curves.mean(axis=0)
 for curve in condition_curves:ax.plot(x,curve,alpha=.65,lw=.9)
 boot=np.random.default_rng(20261101).integers(0,8,(2000,8))
 ci=np.quantile(condition_curves[boot].mean(axis=1),[.025,.975],axis=0)
 ax.fill_between(x,q1,q3,color='steelblue',alpha=.22,label=f'IQR across {n} trajectories')
 ax.plot(x,mean,color='black',lw=2,label=f'Equal-condition mean ({seed_label} seeds each)')
 ax.axvspan(.9,1,color='gold',alpha=.15)
 ax.set(xlabel='Physical noise sigma',ylabel='Turning rate (rad / sigma)',title=branch+' / '+kind,xlim=(0,1),ylim=(0,None))
 ax.legend(fontsize=8)
 lower=np.array([r['sigma'] for r in rows]);upper=np.array([r['sigma_previous'] for r in rows]);width=upper-lower
 high_width=np.maximum(0,upper-np.maximum(lower,.9))
 prior=mean**2
 mass_high=float(np.sum(prior*high_width)/np.sum(prior*width))
 low_snr_width=np.maximum(0,upper-np.maximum(lower,.5))
 # Common measured support only: not a claimed reproduction of paper's full-support 95.69%.
 summary['metrics'][branch+'/'+kind]={
  'sigma_measured_min':float(lower.min()),'sigma_measured_max':float(upper.max()),
  'mean_curve_peak_sigma':float(x[mean.argmax()]),
  'per_sample_peak_sigma':x[matrix.argmax(axis=1)].tolist(),
  'per_condition_mean_peak_sigma':x[condition_curves.argmax(axis=1)].tolist(),
  'prefix_mean_peak_sigma':{str(k):float(x[matrix[:k].mean(axis=0).argmax()]) for k in [8,32,64,128] if k<=n},
  'prefix_mean_curves':{str(k):matrix[:k].mean(axis=0).tolist() for k in [8,32,64,128] if k<=n},
  'sigma_mid_grid':x.tolist(),
  'condition_bootstrap_mean_95ci':ci.tolist(),
  'condition_bootstrap_note':f'Resample 8 fixed conditions with their {seed_label} seeds grouped; descriptive sensitivity, not random sampling of the full benchmark.',
  'unsmoothed_squared_mean_prior_mass_sigma_ge_0_9_measured_support':mass_high,
  'unsmoothed_squared_mean_prior_mass_sigma_gt_0_5_measured_support':float(np.sum(prior*low_snr_width)/np.sum(prior*width)),
  'high_region_mean_rate':float(np.sum(mean*high_width)/high_width.sum()),
  'rest_region_mean_rate':float(np.sum(mean*(width-high_width))/(width-high_width).sum())}
 with (out/(branch+'_'+kind+'.csv')).open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['sigma_mid','mean','q25','q75']+[f'sample_{i}' for i in range(n)])
  w.writerows(zip(x,mean,q1,q3,*matrix))
fig.suptitle(f'PF-Wan official sampler: {n} trajectories, 8 conditions, {seed_label} seeds each' + (' - preliminary' if n not in (36,128) else ''))
fig.savefig(out/'turning_profiles.png',dpi=180);plt.close(fig)
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
