import json,statistics,math,sys
from pathlib import Path
import numpy as np
import torch
from diffusers import DDIMScheduler,AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

root=Path('/data5/shenglin.zeng/sdxl_crossdistill');out=root/'outputs/hparam_pilot/teacher_turning'
ac=DDIMScheduler.from_pretrained(root/'models/sdxl-base-1.0',subfolder='scheduler',local_files_only=True).alphas_cumprod.numpy()
alpha=np.sqrt(ac);sigma=np.sqrt(1-ac);u=sigma/(alpha+sigma)
# Limit peaks to measured adjacent-step midpoints; no endpoint extrapolation.
ugrid=np.linspace((u[0]+u[9])/2,(u[999]+u[989])/2,300)
fields=['rf_full','rf_spatial_centered','rf_coarse_centered','ve_epsilon','rf_conditional_on_guided_path']

def smooth(values,bandwidth=.015):
    weights=np.exp(-.5*((ugrid[:,None]-ugrid[None,:])/bandwidth)**2)
    weights/=weights.sum(1,keepdims=True)
    return values@weights.T

def peak(values,mask=None):
    ids=np.arange(len(ugrid)) if mask is None else np.where(mask)[0]
    pos=ids[np.argmax(values[ids])]
    return {'u':float(ugrid[pos]),'native_t':float(np.interp(ugrid[pos],u,np.arange(1000))),'rate':float(values[pos])}

modes={};samples={}
for label in ['base_dt10_bf16','fine_dt5_bf16','precision_dt10_fp32']:
    records=[]
    for rank in range(4):records+=json.loads((out/f'{label}_rank{rank}.json').read_text())['records']
    assert len(records)=={'base_dt10_bf16':32,'fine_dt5_bf16':16,'precision_dt10_fp32':8}[label]
    curves={};summary={}
    for field in fields:
        rates=[];angles=[]
        for rec in records:
            xx=np.array([r['u_mid'] for r in rec['rows']])[::-1]
            yy=np.array([r['fields'][field]['rate_rad_per_u'] for r in rec['rows']])[::-1]
            rates.append(np.interp(ugrid,xx,yy))
            angles.append(sum(r['fields'][field]['angle_rad'] for r in rec['rows']))
        arr=np.asarray(rates);smoothed=smooth(arr);mean=arr.mean(0);sm=smoothed.mean(0)
        q1,q3=np.quantile(arr,[.25,.75],axis=0)
        interior=(ugrid>=u[20])&(ugrid<=u[979]);early=(ugrid>=u[800])&(ugrid<=u[979])
        rng=np.random.default_rng(20261017)
        # Resample prompts with both seeds together, rather than treating seeds as independent prompts.
        by_prompt=[]
        for idx in sorted(set(r['cache_index'] for r in records)):
            by_prompt.append(smoothed[[i for i,r in enumerate(records) if r['cache_index']==idx]].mean(0))
        by_prompt=np.stack(by_prompt);indices=rng.integers(0,len(by_prompt),size=(1000,len(by_prompt)))
        boot=by_prompt[indices].mean(1)
        boot_peaks=[peak(v,early)['native_t'] for v in boot]
        summary[field]={'raw_mean_rate_peak':peak(mean),'smoothed_global_peak':peak(sm),'smoothed_interior_peak':peak(sm,interior),
            'smoothed_high_noise_peak_t800_979':peak(sm,early),'high_noise_peak_95pct_prompt_bootstrap_interval':np.quantile(boot_peaks,[.025,.975]).tolist(),
            'mean_total_turning_radians':float(np.mean(angles)),
            'bandwidth_sensitivity':{str(b):peak(smooth(arr,b).mean(0),early) for b in [.01,.02,.04]},
            'peak_interpretation':'Maximum is at the upper observed/search boundary, not a localized interior optimum. Bootstrap pile-up there does not mean zero uncertainty.'}
        curves[field]={'mean':mean.tolist(),'p25':q1.tolist(),'p75':q3.tolist(),'smoothed_mean':sm.tolist()}
    modes[label]={'trajectories':len(records),'prompts':len(set(r['cache_index'] for r in records)),'fields':summary,'curves':curves}
    samples[label]=records

# Equal 20-native-timestep bins also report unnormalized angle, as requested.
bins=[]
for low in range(0,1000,20):
    values=[];angles=[]
    for rec in samples['base_dt10_bf16']:
        selected=[r for r in rec['rows'] if low<=r['t_mid']<low+20]
        if selected:
            values.append(sum(r['fields']['rf_full']['angle_rad'] for r in selected)/sum(r['delta_u'] for r in selected))
            angles.append(sum(r['fields']['rf_full']['angle_rad'] for r in selected))
    if values:bins.append({'native_t_low':low,'native_t_high':low+20,'mean_rate':float(np.mean(values)),'mean_angle_degrees':float(np.rad2deg(np.mean(angles)))})

comparisons={}
for label in ['fine_dt5_bf16','precision_dt10_fp32']:
    keys={(r['cache_index'],r['seed']) for r in samples[label]}
    base=[r for r in samples['base_dt10_bf16'] if (r['cache_index'],r['seed']) in keys]
    for field in ['rf_full','rf_spatial_centered','rf_coarse_centered']:
        aa=[]
        for rec in base:
            xx=np.array([r['u_mid'] for r in rec['rows']])[::-1]
            yy=np.array([r['fields'][field]['rate_rad_per_u'] for r in rec['rows']])[::-1]
            aa.append(np.interp(ugrid,xx,yy))
        matched=smooth(np.array(aa)).mean(0);control=np.array(modes[label]['curves'][field]['smoothed_mean'])
        comparisons[f'{label}:{field}']={'matched_base_high_noise_peak':peak(matched,(ugrid>=u[800])&(ugrid<=u[979])),
            'control_high_noise_peak':modes[label]['fields'][field]['smoothed_high_noise_peak_t800_979'],
            'smoothed_profile_correlation':float(np.corrcoef(matched,control)[0,1]),'matched_base_curve':matched.tolist(),'control_curve':control.tolist()}

report={'source':'https://arxiv.org/html/2609.31349v1','metric':'DyMD Eq6: acos(unit_velocity_i dot unit_velocity_j) / abs(delta physical RF noise)',
    'adaptation':'Image primary C=identity; temporal centering at T=1 is zero. Spatial centering and coarse pooling are image-specific ablations.',
    'coordinate':'u=sigma_VP/(alpha_VP+sigma_VP); y=x_VP/(alpha_VP+sigma_VP); v_RF=epsilon_hat-x0_hat; same SNR as original VP',
    'teacher_cfg':8,'main_data':'16 held-out LAION prompts, indices96..111, two seeds101/102;32 trajectories',
    'smoothing':'Average across trajectories, Gaussian smoothing bandwidth0.015 in RF-equivalent u; raw profiles and bandwidth sensitivity retained',
    'u_grid':ugrid.tolist(),'native_t_grid':np.interp(ugrid,u,np.arange(1000)).tolist(),'modes':modes,'native_t_bins20':bins,'matched_controls':comparisons,
    'coordinate_at_candidates':{str(t):{'u':float(u[t]),'log_snr':float(np.log(ac[t]/(1-ac[t])))} for t in [999,979,949,919,899,849,749,599]},
    'cautions':['Turning is a local direction-reorganization statistic, not proof that all diversity has been fixed.',
        'DyMD uses its profile for re-noising importance sampling, not as a proven CrossDistill crossover selector.',
        'Native SDXL terminal SNR is nonzero; mapped u does not reach1. Endpoints and interior peaks are both reported.',
        'The global maximum may be late. High-noise peak is separately reported, with its search range explicit.',
        'Precision control uses FP32 weights/activations with TF32 matmul permitted; it reloads the original FP16 checkpoint. Text conditions remain from the shared BF16 cache.',
        'Whole-image rate magnitude changes with solver stride and precision. Coarse-centered profiles are more stable; use broad location evidence, not exact magnitude or a unique crossover.',
        'The primary maximum is at the first observed interval999->989. High-noise bootstrap peaks pile up at the search boundary; a degenerate interval is not evidence of a precisely identified optimum.']}
(out/'result.json').write_text(json.dumps(report,indent=2))
print('TURNING_SUMMARY',json.dumps(modes['base_dt10_bf16']['fields']['rf_full']),flush=True)
if '--statistics-only' in sys.argv:raise SystemExit(0)

# Four earlier candidate clean estimates, from one fixed holdout prompt and seed.
with torch.inference_mode():
    torch.set_num_threads(4);torch.cuda.set_device(0)
    vae=AutoencoderKL.from_pretrained(root/'models/sdxl-base-1.0',subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).to('cuda').eval()
    processor=VaeImageProcessor(vae_scale_factor=8)
    example=torch.load(out/'example_rank0.pt',weights_only=True)
    selected=[979,949,919,899]
    for t in selected:
        x=example['clean_estimates'][t].to('cuda',dtype=torch.float32)
        image=vae.decode(x/vae.config.scaling_factor,return_dict=False)[0]
        processor.postprocess(image,output_type='pil')[0].save(out/f'early_t{t}.png')
    final=vae.decode(example['final'].to('cuda')/vae.config.scaling_factor,return_dict=False)[0]
    processor.postprocess(final,output_type='pil')[0].save(out/'example_final.png')
    (out/'early_example.json').write_text(json.dumps({'prompt':example['prompt'],'cache_index':example['cache_index'],'seed':example['seed'],'native_t':selected,'meaning':'teacher x0 estimates, not finished teacher samples'}))
print('TEACHER_TURNING_REPORT_COMPLETE',flush=True)
