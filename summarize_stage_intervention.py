import json,html,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=Path('output/sdxl-crossdistill/stage_intervention_20261026');raw=out/'raw'
records=sum([json.loads(f.read_text()) for f in sorted(raw.glob('metrics_rank*.json'))],[])
assert len(records)==192
names=['student','replace1','replace2','replace3','replace4','teacher'];ids=list(range(96,128))
lookup={(r['prompt_index'],r['variant']):r for r in records};assert len(lookup)==192
metrics=['dino_distance_to_teacher','lpips_full_to_teacher','lpips_patch_to_teacher','lowfreq_L1_to_teacher','highpass_logratio_to_teacher']
values={m:np.array([[lookup[i,n][m] for n in names] for i in ids]) for m in metrics}
rng=np.random.default_rng(20261026);boots=rng.integers(0,32,(2000,32))
summary={'n_prompts':32,'seed':201,'checkpoint':'uniform joint4 G160','means':{},'paired_improvement_vs_student':{},'change_from_student':{},
         'wall_minutes':json.loads((raw/'complete.json').read_text())['wall_seconds']/60,
         'peak_sampling_GiB':max(json.loads(f.read_text())['peak_sampling_GiB'] for f in raw.glob('worker*_complete.json'))}
for m,v in values.items():
 summary['means'][m]=dict(zip(names,v.mean(0).tolist()))
 delta=v[:,0:1]-v[:,1:5];ci=np.quantile(delta[boots].mean(1),[.025,.975],axis=0)
 summary['paired_improvement_vs_student'][m]={n:{'mean':float(delta[:,j].mean()),'CI95':ci[:,j].tolist(),'fraction_positive':float((delta[:,j]>0).mean())} for j,n in enumerate(names[1:5])}
change_keys=['dino_distance_to_student','lowfreq_L1_to_student','lpips_patch_to_student','highpass_relative_change']
changes={}
for m in change_keys:
 v=np.array([[((lookup[i,n]['highpass_RMS']/lookup[i,'student']['highpass_RMS'])-1) if m=='highpass_relative_change' else lookup[i,n][m] for n in names[1:5]] for i in ids])
 changes[m]=v;ci=np.quantile(v[boots].mean(1),[.025,.975],axis=0)
 summary['change_from_student'][m]={n:{'mean':float(v[:,j].mean()),'CI95':ci[:,j].tolist()} for j,n in enumerate(names[1:5])}
for i in ids:
 assert abs(lookup[i,'teacher']['lpips_full_to_teacher'])<1e-6
 assert abs(lookup[i,'student']['lpips_full_to_student'])<1e-6
for f in raw.glob('worker*_complete.json'):
 assert all(x['noop_max_abs']==0 for x in json.loads(f.read_text())['checks'])
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (out/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,axes=plt.subplots(1,3,figsize=(14,4.8))
for ax,m,label in zip(axes,metrics[:3],['DINO feature distance','Full-image LPIPS','Matched-patch LPIPS']):
 stats=summary['paired_improvement_vs_student'][m];means=np.array([stats[n]['mean'] for n in names[1:5]])
 ci=np.array([stats[n]['CI95'] for n in names[1:5]])
 ax.bar(range(4),means,color=['#208173' if x>0 else '#c47745' for x in means])
 ax.errorbar(range(4),means,yerr=np.maximum(0,np.stack([means-ci[:,0],ci[:,1]-means])),fmt='none',color='#223746',capsize=4)
 ax.axhline(0,color='#555',lw=.8);ax.set_xticks(range(4),['Replace 1','Replace 2','Replace 3','Replace 4'])
 ax.set(title=label,ylabel='Teacher-reference error reduction');ax.grid(axis='y',alpha=.2)
fig.suptitle('Single-stage teacher replacement | Positive = closer to teacher, not necessarily better quality')
fig.tight_layout();fig.savefig(out/'stage_effects.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(1,4,figsize=(15,4.5))
for ax,m,label in zip(axes,change_keys,['DINO feature change','Low-frequency RGB change','Matched-patch LPIPS change','High-pass RMS change']):
 v=changes[m];ci=np.quantile(v[boots].mean(1),[.025,.975],axis=0);mu=v.mean(0)
 ax.bar(range(4),mu,color='#236c94');ax.errorbar(range(4),mu,yerr=np.maximum(0,np.stack([mu-ci[0],ci[1]-mu])),fmt='none',color='#223746',capsize=4)
 ax.set_xticks(range(4),['R1','R2','R3','R4']);ax.set(title=label);ax.axhline(0,color='#555',lw=.8);ax.grid(axis='y',alpha=.2)
fig.suptitle('Change relative to unchanged student | Sensitivity is not quality improvement')
fig.tight_layout();fig.savefig(out/'stage_sensitivity.png',dpi=170);plt.close(fig)
body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>SDXL逐阶段老师替换实验</title><style>
body{font:16px/1.8 "Microsoft YaHei",sans-serif;background:#eef3f6;color:#193348}main{max-width:1400px;margin:24px auto;background:white;padding:30px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px 16px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}details{margin:14px 0;padding:12px;background:#f3f6f8}</style><main>
<h1>逐阶段替换：哪一步的老师修正会改变最终图像？</h1><p>32条缓存提示词（索引96–127），固定seed201。均匀四点末步GAN+DMD的G160学生，student CFG1，原始SDXL teacher CFG8。冻结所有权重，仅对单个时间区间进行老师多步替换，后续仍用同一学生。六组共192张图。</p>
<p><b>评价口径：</b>此报告衡量阶段敏感性和对老师终图的保真度，不把老师逐样本输出当作真实图像质量上限。已有学生经过末步GAN训练，结论可能依赖这个检查点。中间区间结束仍是带噪状态，未直接把预测x0作为下一步输入。</p>
<details><summary>查看每个替换分支的具体计算</summary><p>先缓存未替换学生的四个阶段输入。选择第k段时，从完全相同的学生输入开始，用老师在该区间多步积分到对应边界，再让原学生完成剩余步骤。使用相同提示词和初始噪声；原采样器没有后续随机加噪。完整老师参考独立从同一初始噪声出发。</p><table><tr><th>分支</th><th>老师替换区间</th><th>老师调用次数</th><th>替换后学生调用次数</th></tr><tr><td>replace1</td><td>999→749</td><td>25</td><td>3</td></tr><tr><td>replace2</td><td>749→499</td><td>25</td><td>2</td></tr><tr><td>replace3</td><td>499→249</td><td>25</td><td>1</td></tr><tr><td>replace4</td><td>249→干净图</td><td>26</td><td>0</td></tr></table><p>老师单次调用的CFG正负条件合批；此处调用次数不等同于条件分支计算量。老师最后经过索引0再到干净边界−1。GPU无替换控制要求重算学生第2步及其后续与原学生输出逐位相等，CPU解析测试检查带噪边界及终点。</p></details>
<h2>首先看改变了什么</h2><img src="stage_sensitivity.png"><p>相对未替换学生的变化：DINO、低频RGB、局部LPIPS及高频强度。高频强度变化为相对比例（−0.2表示下降20%）。这些是敏感性指标；不同列的绝对数值不可直接比较，变化大也不等于质量更好。</p>
<h2>再看是否更接近老师</h2><img src="stage_effects.png"><p>误差下降为正表示更接近老师。误差条是32条提示词配对bootstrap的逐项95%区间（2000次），未作多重比较校正。小样本诊断不计算FID/Patch FID，也不据此宣称GAN的质量收益。</p>
<h2>各组对老师参考的平均距离</h2><table><tr><th>组别</th><th>DINO距离↓</th><th>全图LPIPS↓</th><th>局部LPIPS↓</th><th>低频RGB L1↓</th></tr>'''
for j,n in enumerate(names):body+='<tr><td>'+n+'</td>'+''.join(f'<td>{values[m][:,j].mean():.5f}</td>' for m in metrics[:4])+'</tr>'
body+='''</table><h2>固定顺序的全部对照</h2><p>每组从左至右为原学生、替换1/2/3/4、全老师。上排是完整图缩略，下排是同一位置的256×256原始像素局部（左上角128,128），使用无损PNG。位置固定，可能不覆盖主体。LPIPS局部分数使用四个象限的固定对应裁剪；高频强度仅作为描述，不是质量分数。</p>'''
for i in ids:
 case=json.loads((raw/f'case_{i:03d}.json').read_text())
 links=' · '.join(f'<a href="raw/full/{i:03d}_{n}.png">{n}原图</a>' for n in names) if i<100 else '全分辨率图保存在服务器stage_intervention目录'
 body+=f'<details {"open" if i==96 else ""}><summary>提示词{i}: {html.escape(case["prompt"])}</summary><img src="raw/gallery/{i:03d}.png"><p>{links}</p></details>'
body+='''<p>DINOv2-small使用224输入和ImageNet归一化；全图LPIPS-squeeze使用512输入，局部LPIPS使用原始256裁剪。高频量为Laplacian RMS，不能区分有意义的纹理与噪声。更接近老师与视觉质量变好需要分开判断；替换有害也可能来自老师边界与学生后续的适配问题。</p>
<p><a href="protocol.json">实验方案</a> · <a href="cpu_check.json">CPU边界检查</a> · <a href="summary.json">统计结果</a> · <a href="metrics.csv">逐图指标</a></p><pre>'''+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre></main></html>'
findings=f'''<h2>本次得到的主要信号</h2><ul>
<li>第1步替换对全局外观/构图影响最大：与老师的DINO距离从{values[metrics[0]][:,0].mean():.3f}降至{values[metrics[0]][:,1].mean():.3f}，全图LPIPS从{values[metrics[1]][:,0].mean():.3f}降至{values[metrics[1]][:,1].mean():.3f}。这是更接近同种子老师，不能等同于真实质量提升。</li>
<li>第4步替换对高频响应有明显作用：Laplacian RMS平均变化{changes['highpass_relative_change'][:,3].mean()*100:.1f}%，而全图LPIPS误差下降的95%区间跨0。它改变的内容与第1步不同，但尚不能称为独立、纯粹的纹理阶段。</li>
<li>固定样例97（山景）与98（人物/树叶）中，可见原学生及部分替换分支有网格/条纹状伪纹理，末步老师替换后减轻。说明高频更强未必更好。此观察来自查看样例，不是32幅图的人工盲评统计，也不能确定伪纹理由GAN造成。</li>
<li>第3步替换使平均全图/局部LPIPS更远离老师。可能涉及教师边界与学生后续的适配，但本实验未验证原因；不能据此判定第3步不需要监督。</li></ul>
<p><b>对后续设计的含义：</b>目前更值得追踪“末步伪纹理/过度高频”的变化，而非继续寻找老师转向峰。仍需同采样协议的无GAN学生或GAN开关消融，才能判断GAN的具体贡献；不要据此直接固定TM/DMD/GAN的最优分界。</p>
<p>正式32条诊断耗时{summary['wall_minutes']:.1f}分钟（另有1条短测），采样峰值显存{summary['peak_sampling_GiB']:.2f} GiB。32条均通过无操作控制，192张图及评分完整；实验已退出。</p>'''
body=body.replace('<h2>首先看改变了什么</h2>',findings+'<h2>首先看改变了什么</h2>')
(out/'report.html').write_text(body,encoding='utf-8');print(json.dumps(summary,indent=2))
