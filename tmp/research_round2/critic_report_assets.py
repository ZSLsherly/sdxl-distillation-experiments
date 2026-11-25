"""Build report-ready figure/table from saved experiment results."""
from pathlib import Path
import json
from PIL import Image,ImageDraw,ImageFont
R=Path(__file__).resolve().parent
rbf=json.loads((R/'critic_learned_rbf_results.json').read_text())
multi=json.loads((R/'critic_multimodal_results.json').read_text())
near=json.loads((R/'critic_near_convergence_results.json').read_text())
entropy=json.loads((R/'critic_entropy_direction_results.json').read_text())

learned=[]
for fit,label in [('population_ridge','容量受限，精确训练期望'),('sample_DSM','4,096 个 DSM 样本')]:
 for method,mlabel in [('extrapolation','简单外推'),('transport_oracle_drift','输运（oracle 漂移）'),('fresh_current_DSM_fit','当前分布重新拟合')]:
  s=rbf['summary'][fit][method]
  learned.append(dict(setting=label,method=mlabel,cases=rbf['summary'][fit]['num_cases'],
    median_score_error_ratio=s['median_score_ratio_vs_stale'],
    median_batch32_gradient_error_ratio=s['median_gradient32_ratio_vs_stale']))
high=[r for r in multi['selected_cases'] if r['scenario']=='translation' and r['noise']==.02 and r['delta']==.1 and r['error']=='sin_shared_80']
nc=near['summary']['gap=0.01,error=constant_bias']
gauss=next(r for r in entropy['gaussian'] if r['noise']==.1)
tables={
 'recommended_status':'降为诊断 / 高风险备选，不作为当前算法主线。',
 'learned_critic':learned,
 'high_frequency_representative':[dict(method=r['method'],score_rmse=r['score_rmse'],batch32_gradient_error=r['gradient_error_rmse_batch32']) for r in high],
 'near_target_constant_bias':[dict(method=m,**nc[m]) for m in ['stale','ema','extrapolation','transport_oracle_drift','current_true_score']],
 'hidden_weight_counterexample':gauss,
 'selection_rules':[
  'Learned-critic panel aggregates the full prespecified 48 population-RBF or 240 sample-DSM cases; median is over ratios to each matching stale baseline.',
  'Derivative-noise panel uses the transparent translation case tau=0.02, delta=0.1, sinusoidal score error omega=80, amplitude=10% of old-score RMS; it is not claimed to be representative of real networks.',
  'Near-target panel uses all 32 cases with target path gap=0.01 and constant critic bias=10% of old-score RMS, fixed LR=0.01, norm cap=0.1.',
  'Conditional-weight panel is the exact scalar Gaussian X=theta Z+tau epsilon with theta=1 and tau=0.1.'
 ],
 'information_and_cost_limits':[
  'Transport gets exact conditional drift and its derivatives; this is an oracle favorable to transport.',
  'Extrapolation receives two historical score snapshots and a straight/equispaced parameter history; this favors extrapolation.',
  'No equal-FLOPs or equal-GPU-hour comparison has been run.',
  'Fresh-current RBF is a complete closed-form refit, not one gradient step.',
  'Batch-32 gradient-error RMSE is calculated from exact first/second moments of critic-induced parameter error; no stochastic optimizer or real diffusion model is trained.',
  'These figures measure surrogate score/gradient errors. They do not establish two-step image/video quality or inference acceleration.'
 ]}
(R/'critic_table.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2),encoding='utf-8')

im=Image.new('RGB',(1800,1320),'white');d=ImageDraw.Draw(im)
font=lambda n:ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',n)
ink='#152b3e';muted='#516678';gray='#7e8d9a';teal='#087f8c';orange='#cf7437';grid='#dce4ea';red='#b54e4a'
def text(x,y,s,n=26,color=ink,anchor=None):d.text((x,y),s,font=font(n),fill=color,anchor=anchor)
def barpanel(x,y,w,h,labels,values,colors,ymax,yticks,fmt='{:.3f}'):
 for v in yticks:
  yy=y+h-v/ymax*h;d.line((x,yy,x+w,yy),fill=grid,width=2);text(x-12,yy,f'{v:g}',24,muted,'rm')
 d.line((x,y,x,y+h,x+w,y+h),fill=muted,width=2)
 for i,(lab,v,col) in enumerate(zip(labels,values,colors)):
  xx=x+w*(i+.5)/len(labels);bw=min(106,w/len(labels)*.55);yy=y+h-v/ymax*h
  d.rectangle((xx-bw/2,yy,xx+bw/2,y+h),fill=col)
  text(xx,yy-36,fmt.format(v),26,col,'mt');text(xx,y+h+18,lab,25,muted,'mt')
text(65,40,'critic 输运：从“公式成立”到“研究值得做”，证据仍有缺口',37)
text(65,100,'CPU 多峰解析 + 学得 RBF critic；尚无图像 / 视频训练，也没有等算力优势结论。',25,muted)
text(70,174,'A  学得 critic 后，没有稳定超过简单外推',30)
text(70,221,'有限 batch（32）参数梯度误差 / stale；越低越好',25,muted)
x,y,w,h=130,302,650,215
vals=[]
for fit in ['population_ridge','sample_DSM']:
 vals.extend([rbf['summary'][fit][m]['median_gradient32_ratio_vs_stale'] for m in ['extrapolation','transport_oracle_drift']])
barpanel(x,y,w,h,['外推','输运','外推','输运'],vals,[teal,orange,teal,orange],.6,[0,.2,.4,.6])
text(292,575,'容量受限：48 个设置',24,muted,'mt');text(617,575,'样本 DSM：240 个设置',24,muted,'mt')
text(70,622,'输运仍获得 oracle 漂移；这些不是等训练成本结果。',24,muted)

text(975,174,'B  输入导数会放大高频 critic 误差',30)
text(975,221,'同一平移例：有限 batch（32）参数梯度误差',25,muted)
methods=['stale','extrapolation','transport_oracle_drift']
hv=[next(r['gradient_error_rmse_batch32'] for r in high if r['method']==m) for m in methods]
barpanel(1040,302,650,215,['stale','简单外推','输运'],hv,[gray,teal,orange],1.1,[0,.5,1.])
text(975,575,'τ=0.02，δ=0.1，ω=80；误差幅度为 10%。',24,muted)
text(975,622,'高频误差可在总体梯度中抵消，但有限 batch 方差仍大。',24,muted)

text(70,712,'C  接近收敛：旧 critic 偏差仍会造成错误更新',30)
text(70,759,'32 个设置中，一次参数更新使真实 KL 升高的个数',25,muted)
cv=[nc[m]['kl_increase_count'] for m in ['stale','extrapolation','transport_oracle_drift','current_true_score']]
barpanel(130,840,650,215,['stale','简单外推','输运','真实 score'],cv,[gray,teal,orange,'#a5b9ac'],16,[0,8,16],fmt='{:.0f}')
text(70,1115,'目标与学生很近，旧 critic 加恒定偏差；固定学习率。',24,muted)
text(70,1162,'这否定“预测即可保证改进”，不等于所有更新都会变坏。',24,muted)

text(975,712,'D  替代方向先要避开隐藏变量偏差',30)
text(975,759,'标量高斯：负熵对学生尺度 θ 的真实方向导数',25,muted)
text(1005,840,'真实值 / 条件切向恒等式',27);text(1670,840,f"{gauss['true_negentropy_derivative']:.3f}",30,teal,'rt')
d.line((1005,895,1670,895),fill=grid,width=2)
text(1005,924,'直接用单样本噪声标签回传',27);text(1670,924,'0.000',30,gray,'rt')
d.line((1005,979,1670,979),fill=grid,width=2)
text(1005,1008,'以同一样本的 J² 加权 DSM',27);text(1670,1008,f"{gauss['hidden_weight_Z2_limit']:+.3f}",30,red,'rt')
text(975,1115,'后一项连符号都错；必须先条件化 m(x)=E[Ju|x]。',24,muted)
text(975,1162,'直接熵方向估计仍是待验证线索，尚未证明可扩展。',24,muted)

d.line((65,1230,1735,1230),fill=grid,width=2)
text(65,1257,'结论：A 降为备选。只有真实漂移可低成本预测，且胜过等成本 DSM / 外推，才重新升级。',26,ink)
im.save(R/'critic_round2_summary.png')
print('Wrote critic_round2_summary.png and critic_table.json')
