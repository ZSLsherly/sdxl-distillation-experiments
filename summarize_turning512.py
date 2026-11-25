"""Summarize the completed 128-prompt, four-seed SDXL teacher study."""
import json,csv,html
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('output/sdxl-crossdistill/teacher_turning_resample_20261026/results512')
raw=root/'raw';data=[json.loads(f.read_text()) for f in sorted(raw.glob('base_dt10_bf16_*.json'))]
assert len(data)==512 and len({(d['prompt_index'],d['seed']) for d in data})==512
t=np.array([r['t_mid'] for r in data[0]['rows']]);u=np.array([r['u_mid'] for r in data[0]['rows']])
du=np.array([r['delta_u'] for r in data[0]['rows']])
metrics=['angle_rad','rate_rad_per_u','rate_rad_per_path_rms']
labels=['Angle per solver interval (rad)','Turning / physical noise u','Turning / RMS path distance']
arrays={m:np.array([[r['fields']['rf_full'][m] for r in d['rows']] for d in data]) for m in metrics}
dist=np.array([[r['fields']['rf_full']['path_rms_distance'] for r in d['rows']] for d in data])
assert all(np.isfinite(v).all() for v in arrays.values())
idx=np.random.default_rng(20261026).integers(0,128,(1000,128));bands={}
for m,arr in arrays.items():
 by_prompt=np.stack([arr[[i for i,d in enumerate(data) if d['prompt_index']==p]].mean(0) for p in range(128)])
 bands[m]=np.quantile(by_prompt[idx].mean(1),[.025,.975],axis=0)
summary={'trajectories':512,'unique_prompts':128,'seeds':[201,202,203,204],
 'wall_minutes':json.loads((raw/'complete.json').read_text())['wall_seconds']/60,
 'peak_allocated_GiB':max(d['peak_allocated_GiB'] for d in data),'by_metric':{},'regions':{}}
for m,arr in arrays.items():
 peaks=t[arr.argmax(1)];rho=np.array([spearmanr(u,v).statistic for v in arr])
 summary['by_metric'][m]={'mean_curve_peak_native_t':float(t[arr.mean(0).argmax()]),
  'fraction_individual_peaks_t_ge_950':float((peaks>=950).mean()),
  'fraction_individual_peaks_t_le_50':float((peaks<=50).mean()),
  'median_spearman_with_noise':float(np.median(rho)),
  'fraction_spearman_above_08':float((rho>.8).mean())}
for name,lo,hi in [('early_800_980',800,980),('middle_400_600',400,600),('late_20_200',20,200)]:
 mask=(t>=lo)&(t<hi);ang=arrays['angle_rad'][:,mask].sum(1)
 summary['regions'][name]={'angle_per_u':float((ang/du[mask].sum()).mean()),
                          'angle_per_path_RMS':float((ang/dist[:,mask].sum(1)).mean())}
mon=[json.loads(s) for s in (raw/'monitor.jsonl').read_text().splitlines() if s.strip()]
summary['min_monitored_free_GiB']={str(g):min(x['gpus'][str(g)]['free'] for x in mon)/1024 for g in [0,1,4]}
summary['yielding_monitor_samples']={str(g):sum(x['gpus'][str(g)]['state']=='yielding_to_other_task' for x in mon) for g in [0,1,4]}
summary['completed_local_time']='2026-10-26 '+__import__('datetime').datetime.fromtimestamp(mon[-1]['time']).strftime('%H:%M:%S')
(root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(15,8))
for col,(m,label) in enumerate(zip(metrics,labels)):
 arr=arrays[m];low,high=bands[m]
 for row in [0,1]:
  ax=axes[row,col];mask=np.ones(len(t),bool) if row==0 else (t<=950)&(t>=50)
  ax.plot(t[mask],arr.mean(0)[mask],color='#126b90',lw=2,label='Mean of 512 trajectories')
  ax.fill_between(t[mask],low[mask],high[mask],color='#126b90',alpha=.2,label='95% prompt-cluster CI')
  ax.set(xlabel='Native timestep (early -> late)',title=label if row==0 else 'Interior: t=950 to 50')
  ax.invert_xaxis();ax.grid(alpha=.2)
  if row==0:ax.legend(fontsize=8)
fig.suptitle('SDXL: 128 prompts x 4 seeds | CFG 8 | BF16 | 1024px | 101 teacher evaluations',fontsize=14)
fig.tight_layout();fig.savefig(root/'turning512.png',dpi=170);plt.close(fig)
with (root/'curves.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['native_t','u']+[f'{m}_{k}' for m in metrics for k in ['mean','CI_low','CI_high']])
 for j in range(len(t)):w.writerow([t[j],u[j]]+[v for m in metrics for v in [arrays[m][:,j].mean(),bands[m][0,j],bands[m][1,j]]])
body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>SDXL 512条教师转向采样</title>
<style>body{font:16px/1.8 "Microsoft YaHei",sans-serif;background:#f2f5f8;color:#193348}main{max-width:1250px;margin:32px auto;padding:32px;background:white}img{width:100%}pre{background:#edf3f6;padding:18px;white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px 18px;border-bottom:1px solid #ddd}</style><main>
<h1>SDXL：512条教师转向轨迹</h1><p>128条不同提示词 × 4个种子（201–204），原始冻结SDXL，CFG8，1024分辨率。3090 0/1/4三卡、每卡批量8。所有512条均完成，数值有限、无重复提示词/种子组合。</p>
<h2>主要结果</h2><p>这批样本支持“归一化转向总体开头高、后期低”的判断。按路径距离归一化后，早／中／晚区间均值为36.64／10.64／6.13；单轨迹转向率与噪声坐标的Spearman相关系数中位数为0.992，512条均大于0.8。这是强烈的总体趋势，不代表逐点严格单调。</p><p>原始转角在尾部有反弹，而两种归一化后整体下降。因此，不能把原始转角的尾部上升直接解释成“末段单位距离的转向更强”。目前不足以据此定位独特的中段或GAN阶段分界；最多提供高噪声阶段应受到更多关注的候选依据，仍需与普通高噪声偏置作训练消融。</p>
<p>置信区间按128条提示词分组bootstrap，保留同一提示词的四个种子；不把512条轨迹当作512条独立提示词。上排为全程，下排去掉开头与结尾的边界区间，以便观察内部形状。</p>
<img src="turning512.png"><h2>指标口径</h2><p>u=σ/(α+σ)，y=x/(α+σ)，v=ε̂−x̂₀。分别统计相邻速度向量夹角、夹角/Δu、夹角/RMS(Δy)。路径距离的绝对值依赖latent坐标尺度。下表用区间内总转角除以总Δu或总路径距离，再跨轨迹取平均；排除极端边界点。</p><table><tr><th>区间</th><th>转角/Δu</th><th>转角/路径RMS距离</th></tr>'''
for name,v in summary['regions'].items():body+=f'<tr><td>{name}</td><td>{v["angle_per_u"]:.3f}</td><td>{v["angle_per_path_RMS"]:.3f}</td></tr>'
body+='''</table><h2>解释边界</h2><p>这是同一SDXL、CFG8、BF16及固定步长下的扩样结果。本次512条没有包含新一轮FP32或更细步长对照，也没有SD3结果。批量8短测的首步guided ε与批量1相对RMS差异约1.94%，故不能宣称逐位保持旧输出。</p><p>边界处的转向最大值不能直接确定TM、DMD、GAN的最佳训练分界；是否优于普通高噪声偏置，仍需训练消融。</p><p><a href="summary.json">完整统计</a> · <a href="curves.csv">曲线数据</a> · <a href="raw/validation_final.json">完整性验证</a></p><pre>'''+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></main></html>'
(root/'report.html').write_text(body,encoding='utf-8')
print(json.dumps(summary,indent=2))
