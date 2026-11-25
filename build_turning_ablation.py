"""Matched raw-angle / noise-normalized turning priors from the SAME 512 trajectories."""
import hashlib,json,html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path(__file__).resolve().parent
source=base/'output/sdxl-crossdistill/teacher_turning_resample_20261026/results512/raw'
out=base/'output/sdxl-crossdistill/turning_ablation_20261027';out.mkdir(exist_ok=True)
profiles=base/'sdxl_research/turning_profiles';profiles.mkdir(exist_ok=True)
files=sorted(source.glob('base_dt10_bf16_*.json'));assert len(files)==512
h=hashlib.sha256();data=[]
for f in files:
 raw=f.read_bytes();h.update(f.name.encode());h.update(raw);data.append(json.loads(raw))
rows=data[0]['rows'];u=np.array([r['u_mid'] for r in rows]);t=np.array([r['t_mid'] for r in rows])
assert len({(d['prompt_index'],d['seed']) for d in data})==512
for d in data:assert np.array_equal([r['u_mid'] for r in d['rows']],u)
angles=np.array([[r['fields']['rf_full']['angle_rad'] for r in d['rows']] for d in data])
du=np.array([r['delta_u'] for r in rows]);order=np.argsort(u)
ugrid=np.linspace(u.min(),u.max(),256);bandwidth=.015
kernel=np.exp(-.5*((ugrid[:,None]-ugrid[None,:])/bandwidth)**2);kernel/=kernel.sum(1,keepdims=True)
times=np.arange(20,920);u_at_t=np.interp(times,t[::-1],u[::-1]);beta=.3
curves={'normalized_u':(angles/du).mean(0),'raw_angle':angles.mean(0)}
summary={};ps={}
for name,curve in curves.items():
 uniform_curve=np.interp(ugrid,u[order],curve[order]);smooth=kernel@uniform_curve
 rates=np.interp(u_at_t,ugrid,smooth);offset=.1*np.median(rates)
 weight=(rates+offset)**2;prior=weight/weight.sum();q=(1-beta)/900+beta*prior
 p={'name':name,'metric':'theta / abs(delta_u)' if name=='normalized_u' else 'theta (no noise/path division)',
    'beta':beta,'support':[20,919],'timesteps':times.tolist(),'probabilities':q.tolist(),
    'prior_probabilities':prior.tolist(),'profile_values':rates.tolist(),'offset':float(offset),
    'offset_rule':'0.1 * median of each smoothed profile on [20,919]', 'exponent':2,
    'source_sha256':h.hexdigest(),'trajectories':512,'prompts':128,'seeds':[201,202,203,204],
    'smoothing':'Interpolate onto 256 uniform physical-u points, Gaussian bandwidth .015, then interpolate onto native integer t; same for both arms',
    'importance_correction':False,'state_field':'rf_full, C=identity, CFG8, BF16, native stride10',
    'probability_measure':'mass on integer native timesteps; not a density uniform in u'}
 (profiles/f'{name}_512.json').write_text(json.dumps(p,indent=2),encoding='utf-8');ps[name]=p
 summary[name]={'mean_t':float(times@q),'mass_low_20_199':float(q[times<200].sum()),
     'mass_middle_200_799':float(q[(times>=200)&(times<800)].sum()),'mass_high_800_919':float(q[times>=800].sum()),
     'max_to_uniform_ratio':float(q.max()*900),'probability_sum':float(q.sum())}
cfg=json.loads((base/'sdxl_research/configs/boundary_dmd_10k_g160_1gpu.json').read_text())
for name in curves:
 config={**cfg,'turning_profile':f'sdxl_research/turning_profiles/{name}_512.json'}
 (base/f'sdxl_research/configs/turning_{name}_g160.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
protocol={'status':'prepared_not_trained','comparison':summary,'beta':beta,'only_arm_difference':'turning metric before identical smoothing and probability construction',
          'old_C':'prepared from theta/delta_u on 32 trajectories; no completed training record found on source server',
          'common_training':cfg,'source_sha256':h.hexdigest(),
          'scope':'boundary student DM re-noise only; fake uniform in each existing phase range; PCM unchanged; GAN disabled. Fixed prior, NOT full adaptive DyMD.',
          'normalization':'normalized means theta/delta_u; both arms necessarily normalize final probability mass to sum 1; theta/path_distance is a different experiment',
          'required_comparison':'fresh identical SDXL initialization per arm; do not warm-start one arm from the other',
          'compute':'same 160 G updates, global batch64, data order, seeds, microbatch and GPU topology; save G40/80/120/160 and final resumable state'}
import sys
sys.path.insert(0,str(base/'sdxl_research'))
from turning_sampling import TurningSampler
ranges={2:(509,739),3:(259,489),4:(20,239)}
conditional={}
fig,axes=plt.subplots(1,3,figsize=(14,4.3))
for ax,(stage,bounds) in zip(axes,ranges.items()):
 conditional[stage]={}
 for name in curves:
  sampler=TurningSampler(profiles/f'{name}_512.json');ts,qs=sampler.distribution(bounds)
  cut=(bounds[0]+bounds[1])/2
  conditional[stage][name]={'range':bounds,'mean_t':sum(t*q for t,q in zip(ts,qs)),
       'mass_upper_half':sum(q for t,q in zip(ts,qs) if t>cut),'probability_sum':sum(qs)}
  ax.plot(ts,np.array(qs)*len(ts),label=name)
 ax.axhline(1,color='gray',ls='--');ax.invert_xaxis();ax.set(title=f'Stage {stage}: t in {bounds}',xlabel='Re-noise t',ylabel='Probability / stage-uniform');ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'boundary_sampling_comparison.png',dpi=170);plt.close(fig)
protocol['stage_conditional_distributions']=conditional
protocol['phase_selection']='Each G: PCM + one random middle stage 2/3 + terminal4; fake updates twice, cycles 2/3/4 unchanged'
protocol['conditional_mixture']='Recompute offset and normalized prior inside each legal interval, then mix .7 uniform + .3 prior; do not simply truncate the global mixture'
(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for name,p in ps.items():
 values=np.array(p['profile_values']);axes[0].plot(times,values/values.max(),label=name)
 axes[1].plot(times,np.array(p['probabilities'])*900,label=name)
axes[0].set(title='Prior shape (scaled by its own maximum)',ylabel='Relative profile')
axes[1].set(title='Actual student DMD sampling allocation',ylabel='Probability / uniform probability');axes[1].axhline(1,color='gray',ls='--',label='Uniform')
for ax in axes:ax.set_xlabel('Native timestep (early -> late)');ax.invert_xaxis();ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(out/'sampling_comparison.png',dpi=170);plt.close(fig)
body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>归一化与原始转角消融</title><style>body{font:16px/1.8 "Microsoft YaHei",sans-serif;background:#f1f5f8;color:#20384b}main{max-width:1100px;margin:30px auto;background:white;padding:30px}img{max-width:100%}td,th{padding:8px 16px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style><main>
<h1>最新带噪边界DM：归一化转向率 vs 原始转角</h1><p><b>状态：代码与分布准备，尚未训练。</b>按用户确认的最新方案：纯DMD、每次PCM＋随机一个中段DM＋末步DM，fake每G更新两次；GAN关闭。此前C方案准备的是θ/Δu，未找到完成训练记录；已完成的G160、512条轨迹采样与阶段替换实验不能当作已训练的C组。</p>
<p>本次以同一512条教师轨迹构造两组：A使用θ/|Δu|，B使用θ；u=σVP/(αVP+σVP)。两组同样在物理u坐标平滑，再映射到离散原生t∈[20,919]；p∝(曲线+0.1×曲线中位数)²，q=0.7×Uniform+0.3×p，不做重要性补偿。未归一化指“不除以间隔”，不是概率和不等于1。</p>
<h2>实际训练使用的三个条件分布</h2><img src="boundary_sampling_comparison.png"><p>第2步匹配x499，其监督t∈[509,739]；第3步匹配x249，t∈[259,489]；末步匹配干净终点，t∈[20,239]。每个范围内分别重建70%均匀＋30%先验。不会增加首步PCM次数，也不会把中段监督采到非法噪声水平。fake仍按原方案在对应区间均匀采样。</p>
<h2>全局剖面，仅供理解，不是本次实际训练分布</h2><img src="sampling_comparison.png"><table><tr><th>组别</th><th>t&lt;200</th><th>200≤t&lt;800</th><th>t≥800</th><th>平均t</th></tr>'''
for name,v in summary.items():body+=f'<tr><td>{name}</td><td>{v["mass_low_20_199"]:.1%}</td><td>{v["mass_middle_200_799"]:.1%}</td><td>{v["mass_high_800_919"]:.1%}</td><td>{v["mean_t"]:.1f}</td></tr>'
body+='''<tr><td>均匀参考</td><td>20.0%</td><td>66.7%</td><td>13.3%</td><td>469.5</td></tr></table>
<p>训练采用均匀四点999/749/499/249、10K旧数据、batch64、G160；student lr=1e-6，fake lr=5e-6，PCM权重1，梯度裁剪1。只替换各student DM分支的加噪t，原本的随机t仍照常生成后再覆盖，从而保持初始噪声、再加噪epsilon、PCM噪声和裁剪/翻转随机流不变。fake更新两次，仍用各阶段原有均匀噪声；第一步PCM不加权。两组使用相同新代码、初始化与资源配置，独立模型和优化器。</p>
<p>这里“加强前期”是提高高噪声DMD监督采样概率，不是增加四步生成中第1步的更新次数。原始转角依赖采样网格，结论仅适用于固定的native步长10。旧32条剖面不与新512条另一组混用。</p>
<p>这是固定先验消融，不是完整DyMD复现：未加入视频时序亲和度自适应混合和fake难度加权。<a href="https://arxiv.org/html/2609.31349v1#S3.S2">DyMD公式6和10</a></p>
<p><a href="protocol.json">完整配置及概率统计</a> · <a href="checks.json">CPU及静态检查</a></p><pre>'''+html.escape(json.dumps(protocol,ensure_ascii=False,indent=2))+'</pre></main></html>'
(out/'report.html').write_text(body,encoding='utf-8')
print(json.dumps(summary,indent=2))
