from pathlib import Path
import json,html,sys,csv
root=Path(__file__).resolve().parents[1];out=root/'output/sdxl-crossdistill/coco_uniform_g160_20261025'
old=json.loads((out/'previous_scores.json').read_text())
scores=dict(old)
if (out/'scores.json').exists():scores.update(json.loads((out/'scores.json').read_text()))
names={'dmd_only':'纯 DMD · G160','gan4':'末步纯 GAN · G160','joint4':'末步 GAN+DMD · 旧 G160','joint34':'第 3、4 步 GAN+DMD · G160','joint4_uniform':'均匀四点末步 GAN+DMD · 新 G160','dmd2':'官方 DMD2 · 4 步','sdxl':'原始 SDXL · 50 步'}
order=['dmd_only','gan4','joint4','joint34','joint4_uniform','dmd2','sdxl']
chart=''
if scores:
 sys.path.insert(0,str(root/'.plot_runtime_pkgs'))
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 labels=['DMD only','GAN step 4','GAN+DMD step 4','GAN+DMD steps 3+4','Uniform GAN+DMD step 4','Official DMD2 (4 steps)','Original SDXL (50 steps)']
 colors=['#94a3b8']*4+['#2563eb']+['#059669']*2
 fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained',sharey=True)
 for ax,metric in zip(axs,['FID','Patch_FID']):
  values=[scores[k][metric] if k in scores else float('nan') for k in order];ax.barh(labels,values,color=colors)
  for i,(k,v) in enumerate(zip(order,values)):
   if k in scores:ax.text(v+.4,i,f'{v:.3f}',va='center',fontsize=10)
   else:ax.text(1,i,'NOT SCORED - incomplete 10K',va='center',fontsize=10,color='#b45309')
  ax.set_xlim(0,max(s[metric] for s in scores.values())*1.2);ax.set_title(metric.replace('_',' ')+' (lower is better)');ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
 axs[0].invert_yaxis();fig.suptitle('Fixed COCO2014 10K | historical groups + uniform G160')
 fig.savefig(out/'benchmark_comparison.png',dpi=180);fig.savefig(out/'benchmark_comparison.pdf');plt.close(fig)
 chart='<img src="benchmark_comparison.png" alt="COCO10K FID and Patch FID"><p><a href="benchmark_comparison.pdf">下载比较图 PDF</a></p>'
with (out/'combined_scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
 w=csv.writer(f);w.writerow(['method','label','n','FID','Patch_FID'])
 for k in order:
  if k in scores:w.writerow([k,names[k],scores[k]['n'],scores[k]['FID'],scores[k]['Patch_FID']])
(out/'combined_scores.json').write_text(json.dumps(scores,indent=2),encoding='utf-8')
rows=''
for k in order:
 s=scores.get(k);vals=f"<td>{s['n']:,}</td><td>{s['FID']:.4f}</td><td>{s['Patch_FID']:.4f}</td>" if s else '<td>目标 10,000</td><td>运行中，未出分</td><td>运行中，未出分</td>'
 rows+=f'<tr class="{"new" if k=="joint4_uniform" else ""}"><td>{names[k]}</td>{vals}</tr>'
status='新 G160 完整 10K 结果尚未出分。旧六组均为已完成 10K 的结果；原始 SDXL 已更新为完整 10K。'
if 'joint4_uniform' in scores:
 s=scores['joint4_uniform'];b=scores['joint4'];d=scores['dmd_only']
 status=f"新 G160 完成 {s['n']:,} 张。相对旧末步联合组：FID 变化 {s['FID']-b['FID']:+.4f}，Patch FID 变化 {s['Patch_FID']-b['Patch_FID']:+.4f}（负值更好）。相对旧纯 DMD：分别 {s['FID']-d['FID']:+.4f} / {s['Patch_FID']-d['Patch_FID']:+.4f}。"
 status+=f" 相对旧末步联合组，FID 降低 {(1-s['FID']/b['FID'])*100:.2f}%，Patch FID 降低 {(1-s['Patch_FID']/b['Patch_FID'])*100:.2f}%。本次新组在已完成的五组自训 G160 中两项分数均最低，但仍落后于官方 DMD2；采样点、PCM 边界及 fake 阶段策略同时不同，不能把提升单独归因于 GAN。"
progress=''
preview=''
is_stopped=False
if (out/'remote_status.json').exists():
 from datetime import datetime,timezone,timedelta
 p=json.loads((out/'remote_status.json').read_text());workers=p.get('workers',{})
 stamp=datetime.fromtimestamp(p['checked_unix'],timezone(timedelta(hours=8))).strftime('%m-%d %H:%M:%S')
 total=sum(s['done'] for s in workers.values());speed=sum(s['images_per_second'] for s in workers.values())
 progress=f'<p><b>进度快照（北京时间 {stamp}）：已提交特征 {total:,}/10,000 张。</b>'
 if speed:progress+=f' 四进程已报告吞吐合计 {speed:.2f} 张/秒。'
 if not workers:progress+=' 正在加载模型，尚无完整 64 张特征块提交。'
 progress+='</p>'
 events=p.get('guard_events',[]);is_stopped=bool(events and events[-1]['event']=='stopped')
 integrity=[e for e in events if e['event']=='existing_blocks_unchanged']
 if integrity:
  proof=integrity[-1]
  progress+=f"<p><b>复用核验通过：</b>原有 {proof['blocks']} 个特征块、{proof['samples']:,} 张样本的 SHA256 与修改时间均未变化；本次只补缺失部分。</p>"
 if is_stopped and 'joint4_uniform' not in scores:
  status=f'新 G160 尚未完成评测：最后记录为 {total:,}/10,000 张。检测到其他任务进入所用 GPU 后，自己的评测已停止；已有特征块保留。下面六组均为已完成的完整 COCO10K 结果。'
  rows=rows.replace('运行中，未出分','已停止，待补齐')
  progress=f'<p>最后同步：北京时间 {stamp}。停止前已提交 {total:,} 张特征；当时吞吐 {speed:.2f} 张/秒，不代表当前仍在运行。</p>'
  progress+='<p><b>停止原因：</b>'+html.escape(events[-1]['reason'])+'。</p>'
if (out/'manifest.json').exists():
 manifest=json.loads((out/'manifest.json').read_text());items=[]
 for im in sorted((out/'images/joint4_uniform').glob('*.png'))[:4]:
  idx=int(im.stem);prompt=manifest['samples'][idx]['prompt']
  items.append(f'<figure><img src="images/joint4_uniform/{im.name}" alt="sample {idx}"><figcaption>#{idx}: {html.escape(prompt)}</figcaption></figure>')
 if items:preview='<section><h2>固定保留样图预览</h2><p>按原评测预设的保留索引顺序展示前四张，不按观感筛选。少量样图仅用于检查推理输出，不替代完整分数。</p><div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">'+''.join(items)+'</div></section>'
refresh='<meta http-equiv="refresh" content="60">' if 'joint4_uniform' not in scores and not is_stopped else ''
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}<title>均匀四点 G160 · COCO10K 与损失诊断</title>
<style>body{{max-width:1120px;margin:40px auto;padding:0 24px;font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#182c40;background:#f5f7fa}}h1{{font-size:29px}}h2{{margin-top:36px}}section{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:24px;margin:20px 0}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e5e9f0}}.new{{background:#e9f1ff}}.note{{color:#536479}}img{{max-width:100%;background:white}}pre{{font-size:12px;overflow:auto;max-height:400px}}code{{font-size:14px}}</style>
<h1>均匀四点 G160：COCO10K 评测与训练诊断</h1><p class="note">2026-10-25 · 999 / 749 / 499 / 249 · 末步 GAN+DMD · 全局 batch 64</p>
<section><h2>完整评测状态</h2><p>{status}</p><table><tr><th>模型</th><th>样本数</th><th>FID ↓</th><th>Patch FID ↓</th></tr>{rows}</table>
{chart}<p>评测位于 .132.41。首段使用空闲 GPU 1、2、3、6；因其他任务进入 1 号卡而停止后，13:03 改用复查为空闲的 GPU 2、3、5、6，只补剩余 4,304 张。已完成的 89 个特征块（5,696 张）保留，结束时验证其校验值及修改时间均不变。只读取 G160，不修改训练；检测到其他 GPU 进程进入时，停止自己的评测。</p>
<p>沿用 DMD2 固定 COCO2014 10K 文件、提示词顺序与 seed=20261020+行号；1024 分辨率；相同 Inception 权重与预处理。FID 使用中心正方形 → Lanczos 512 → clean 299；Patch FID 使用原图中心 299。保留同一组 128 张图便于视觉复核。</p>
<p class="note">这些是本项目同协议重评结果，不是论文排行榜。Patch FID 是中心裁剪分布距离，并不等于完整的纹理质量或逐图纹理准确率。新旧组还存在采样点、PCM 边界和 fake 训练阶段策略的差异，不能把分数变化单独归因于 GAN。跨 Ampere/Ada 推理保持核心软件版本与协议，但不宣称逐位一致。</p>{progress}</section>
<section><h2>PCM 确实上涨；GAN 并未在后期变大</h2>
<table><tr><th>均值</th><th>早期 G1–40</th><th>后期 G121–160</th><th>解释</th></tr>
<tr><td>PCM 原始损失</td><td>0.000300</td><td>0.000710</td><td>约 2.37 倍</td></tr>
<tr><td>PCM × 10</td><td>0.003002</td><td>0.007102</td><td>实际加权标量</td></tr>
<tr><td>GAN 原始损失</td><td>1.09285</td><td>0.66869</td><td>仅计 GAN 生效的更新</td></tr>
<tr><td>GAN × 0.005274</td><td>0.005764</td><td>0.003527</td><td>实际加权标量</td></tr></table>
<p>PCM 是边界 latent 的均方误差。其 target 使用当前学生的无梯度预测，经老师两小步后构造，并不是固定标签，也不是单独冻结的 EMA 学生；共享 UNet 同时被 DMD 和 GAN 更新。目标移动、采样噪声和多目标牵制，都可能让 PCM 误差增大。<b>这说明高噪声一致性约束拟合变差，不能只用“数值很小”来忽略；但也不等同于训练发散。</b></p>
<p>GAN 采用 hinge generator loss：<code>−mean(D(fake))</code>，分数不是概率，也没有必须趋近零的标准。G111→G120 确有 0.3005→1.0952 的局部反弹，随后回到约 0.8；这是应观察的波动，但整体后期均值没有高于早期。不能与 PCM 的原始值直接比大小；更不能用加权标量比例代替各项的梯度贡献。未启用的两步记录的 GAN=0 也不应混入有效 GAN 损失均值。</p>
<img src="loss_by_stage.png" alt="PCM GAN losses and stage-conditioned gradient norms"><p><a href="loss_by_stage.pdf">下载 PDF 图</a> · <a href="loss_analysis.json">诊断数据</a></p></section>
<section><h2>学生梯度主要受阶段轮换影响</h2>
<p>每次都更新同一个共享学生，但监督阶段按 s=2→3→4 循环。总目标为 <code>L_DMD(s) + 10 L_PCM + 1[s=4] × 0.005274 L_GAN</code>，因此三种更新的梯度尺度本来就不完全相同。</p>
<p>这里的梯度是学生全部参数的合成 L2 范数，在裁剪前记录。后期阶段 2 / 3 / 4 的中位数分别为 <b>0.999 / 0.414 / 0.292</b>；它们按顺序轮换，所以连接成一条折线会产生明显锯齿。G40–160 的分阶段均值差异可解释约 80.7% 的对数梯度方差（描述性统计，不是因果证明）。</p>
<p>160 次更新最大梯度 6.126，95 分位 1.482；裁剪阈值为 10，<b>学生 0/160 次触发裁剪</b>。后期各阶段范围为 0.852–1.231、0.339–0.596、0.214–0.628：阶段 4 的相对波动更大，但绝对梯度仍低于阶段 2。没有观测到梯度持续增长、NaN 或 Inf。</p>
<p>这能排除明显数值爆炸，不能证明优化方向正确。现有日志未记录 PCM、DMD、GAN 的独立参数梯度范数和夹角，无法直接判断相互抵消或哪个目标主导。先看这次独立 COCO10K 结果，再决定是否需要少量固定样本的梯度诊断，当前不改学习率或损失权重。</p></section>
<section><h2>判别器还值得注意什么</h2>
<p>后期 200 次 D 更新的 hinge loss 均值 0.800，真假总评分分别 0.672 / −0.706；没有变成近零损失的完全分离状态。DINO 分支真假评分为 0.262 / −1.948，RGB 分支为 1.082 / 0.537，说明两个分支的分数尺度与区分程度不同。分数不能直接证明哪个分支的梯度主导。</p>
<p>R1 有效更新的原始均值从早期 0.470 增至后期 2.722，说明判别器对输入的敏感程度升高，需结合结果关注。D 自身 800 次更新有 71 次超过裁剪阈值，后 200 次为 6 次；这与上文“学生从未裁剪”是不同统计。没有留出集 D 评分，不能据此排除判别器过拟合。</p></section>
{preview}<section><h2>数据和可复核文件</h2><p>训练：10K LAION 衍生子集；评测：固定 COCO2014 10K，二者并非同一训练数据。G160 为 160 次学生更新、800 次 fake/D 更新，并非 160 个全数据 epoch。</p><p><a href="combined_scores.csv">对照表 CSV</a> · <a href="combined_scores.json">分数 JSON</a> · <a href="../uniform10k_results_20261025/report.html">训练进展与固定样图</a></p></section></html>'''
(out/'report.html').write_text(page,encoding='utf-8');print(out/'report.html')
