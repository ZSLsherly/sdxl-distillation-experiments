from pathlib import Path
import os,sys,json,statistics,html
os.environ['MPLCONFIGDIR']=str(Path('output/sdxl-crossdistill/.matplotlib').resolve())
sys.path.insert(0,str(Path('.plot_runtime_pkgs').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root=Path('output/sdxl-crossdistill/hparam_pilot');assessment=root/'assessment'
report=json.loads((assessment/'result.json').read_text(encoding='utf-8'));protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
turning=json.loads((root/'teacher_turning/result.json').read_text())
turning_main=turning['modes']['base_dt10_bf16']['fields']['rf_full']
global_peak=turning_main['smoothed_interior_peak'];early_peak=turning_main['smoothed_high_noise_peak_t800_979']
trials=[r['name'] for r in protocol['trials']];short=['LR 5e-7 | PCM 10','LR 2.5e-7 | PCM 10','LR 5e-7 | PCM 1','LR 2.5e-7 | PCM 1']
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'#fafafa'})
fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
for trial,label in zip(trials,short):
    rows=[json.loads(line) for line in (root/trial/'rank0.jsonl').read_text().splitlines()]
    students=[r for r in rows if r['phase']=='student']
    axes[0].plot(range(1,len(students)+1),[r['dmd_loss'] for r in students],marker='.',label=label)
    axes[1].plot(range(1,len(students)+1),[r['pcm_loss'] for r in students],marker='.',label=label)
axes[0].set_title('DMD surrogate loss (rank 0; not image quality)');axes[1].set_title('PCM consistency MSE (rank 0)')
for ax in axes:ax.set_xlabel('Student update');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(root/'training_curves.png',dpi=150);plt.close(fig)

for idx in [96,98,100,102]:
    fig,axes=plt.subplots(1,4,figsize=(16,4.7),layout='constrained')
    record=next(r for r in report['records'] if r['cache_index']==idx)
    for ax,trial,label in zip(axes,trials,short):
        ax.imshow(Image.open(assessment/f'{trial}_idx{idx}_seed101.png'));ax.axis('off');ax.set_title(label,fontsize=12)
    prompt=record['prompt'];title=prompt[:170]+('…' if len(prompt)>170 else '')
    fig.suptitle(f'Four short-training configurations | held-out LAION index {idx}, seed 101\n{title}',fontsize=12)
    fig.savefig(root/f'validation{idx}_four_trials.png',dpi=130);plt.close(fig)

fig,axes=plt.subplots(1,4,figsize=(16,4.7),layout='constrained')
for ax,method,label in zip(axes,['teacher_ddim','dmd2','senseflow','lr5e7_pcm10'],['SDXL teacher','Official DMD2 | 4 steps','Official SenseFlow | 4 steps','CrossDistill pilot | 12 student updates']):
    ax.imshow(Image.open(assessment/f'{method}_idx96_seed101.png'));ax.axis('off');ax.set_title(label,fontsize=11)
fig.suptitle('Held-out LAION prompt index 96 | Seed 101 | 1024 px | pretrained references vs short pilot',fontsize=13)
fig.savefig(root/'baseline_four_models.png',dpi=130);plt.close(fig)

metric_rows=[]
for trial,label in zip(trials,short):
    m=report['metrics'][trial];s=m['training'];ci=m['clip_gain_95pct_prompt_bootstrap_ci']
    metric_rows.append(f'<tr><td>{label}</td><td>{s["train_seconds"]/60:.1f} 分</td><td>{s["peak_allocated_GiB"]:.2f} GiB</td><td>{m["boundary_relative_mse_mean"]:.5f}</td><td>{m["clip_cosine_mean"]:.4f}</td><td>{m["clip_gain_vs_untrained"]:+.4f}<br><small>[{ci[0]:+.4f}, {ci[1]:+.4f}]</small></td><td>{m["seed_clip_cosine_distance_mean"]:.4f}</td></tr>')
teacher_sections=[]
names=['棒球场','室内场景','披萨','滑雪']
for rank,name in enumerate(names):
    teacher_sections.append(f'<h3>{name}</h3><a href="teacher_stages/prompt{rank}_four_stages.png"><img src="teacher_stages/prompt{rank}_four_stages.png"></a><p><a href="teacher_stages/prompt{rank}_stages_and_seeds.png">查看四个种子 × 四个阶段</a></p>')
baseline_rows=''.join(f'<tr><td>{name}</td><td>{report["metrics"][name]["clip_cosine_mean"]:.4f}</td><td>{report["metrics"][name]["seed_clip_cosine_distance_mean"]:.4f}</td></tr>' for name in ['base_4step','teacher_ddim','dmd2','senseflow'])
sources=''.join(f'<li><a href="{url}">{html.escape(name)}</a></li>' for name,url in protocol['sources'].items())
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SDXL CrossDistill 少样本试验</title>
<style>body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;color:#182332;background:#fbfcfe}}h1,h2,h3{{line-height:1.3}}h2{{margin-top:42px}}img{{width:100%;border-radius:8px}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #dce3ec}}th{{background:#edf2f8}}.note{{border-left:4px solid #3478c8;background:#edf4fd;padding:16px 20px}}small{{color:#536477}}a{{color:#1763a5}}code{{background:#edf2f8;padding:2px 5px}}</style>
<h1>SDXL CrossDistill 少样本试验</h1><p>2026-10-17 · 172.18.33.60 · 4 × A100 40GB · 1024 × 1024</p>
<div class="note">四组参数均完成 12 次学生更新、60 次 fake score 更新。此轮检验训练稳定性与早期趋势；图像尚未收敛，不能宣布最优参数或超过正式基准。</div>
<p><b>首轮起点建议：</b>学生与 fake score 学习率均为 5e-7，PCM 权重10，fake:student 更新比例5:1。该组在本次留出样本上的 CLIP 为0.2773，初始化四步采样器为0.2434；按提示词重采样的提升95%区间约为[0.0178,0.0509]。但与“仅把学习率减半”或“仅把PCM权重降为1”的组间差异尚不明确，不能据此宣布全局最优。<a href="parameter_comparisons.json">组间配对区间</a></p>
<h2>参考 DyMD：老师在哪里转向</h2><p>主统计使用 16 个独立留出提示词 × 2 个种子，共 32 条老师轨迹。参考 DyMD 第3.2节，计算单位速度相邻夹角除以物理噪声差，再跨轨迹平均和平滑。SDXL 先转换为保持信噪比的 RF 对应坐标。单图使用完整向量，原文的视频时间均值操作在单图上会归零，因此另外展示去空间均值和粗尺度对照。</p><img src="teacher_turning/turning_profile.png"><p><b>最大的实测相邻转向出现在最初的 999→989 区间；按20步合并，980～1000区间也最大。</b>曲线向后逐渐下降，没有得到一个独立、清晰的内部峰。限制在 t800～979 搜索时，最大值仍贴着上边界；重采样置信区间因此堆积在边界，不能解释为精确确定了最佳时间步。全图转向幅度对精度、步长较敏感，粗尺度曲线更稳定。精度对照使用 FP32 权重与激活，允许 TF32 矩阵乘法。详见<a href="teacher_turning/result.json">完整统计</a>。</p><p>这些结果支持下一轮优先比较 <b>949 / 919 / 899</b>。DyMD 本来用转向曲线构造重新加噪的采样先验；它不保证转向最大处就是 CrossDistill 的最佳交接点。</p>
<h3>更早的四个候选位置</h3><img src="teacher_turning/earlier_four_stages.png"><p>依次为 <b>979、949、919、899</b>；同一 LAION 提示词、同一种子，展示老师干净图像预测。根据人工反馈，分界点应重点向更早阶段考察；本轮已排队的优化器对照固定在 749，作为控制组。</p>
<h2>原先四个阶段：每组四张图</h2><p>同一提示词、同一种子，依次为原生时间步 <b>899、749、599、449</b>。显示该状态的老师干净图像预测 x̂₀；这些不是已经完成采样的最终图，也不是带噪状态 xₜ。四个场景、每场景四个种子均保留。</p>
{''.join(teacher_sections)}<h3>完整老师轨迹终点</h3><img src="teacher_stages/teacher_final_references.png">
<h2>四组参数的留出结果</h2><p>可用训练缓存为 96 份，留出 32 份。本轮按固定遍历顺序实际向学生展示了 16 个独立样本、合计 48 次；最终评估使用其中 8 个留出提示词 × 2 个种子，共 16 张图/方法。所有组重新从相同 SDXL 初始化，并使用相同噪声与顺序。</p>
<table><tr><th>参数</th><th>纯训练时间</th><th>PyTorch 峰值/卡</th><th>交接状态相对 MSE ↓</th><th>CLIP ↑</th><th>CLIP 相对初始化变化<br>95% 配对区间</th><th>种子特征距离</th></tr>{''.join(metric_rows)}</table>
<p>CLIP 使用官方 OpenAI ViT-B/32，取未乘尺度的余弦相似度。种子距离大不一定意味着更好，也可能来自噪声与伪影。交接状态误差衡量老师轨迹保持；低噪声 DMD 没有被要求逐像素复制老师终点。此样本量不计算 FID。</p>
<p>服务器有共享负载，第三组末尾还曾与本任务的老师统计短暂重叠；已停止并独立重跑统计。计时用于资源估计，不把组间速度差解释为参数收益。<a href="scheduling_incident.json">事件记录</a> · <a href="audit.json">训练一致性核验</a></p>
<h2>相同留出样本的四组效果</h2>{''.join(f'<img src="validation{idx}_four_trials.png">' for idx in [96,98,100,102])}
<h2>官方模型参考</h2><table><tr><th>模型</th><th>CLIP</th><th>种子特征距离</th></tr>{baseline_rows}</table><img src="baseline_four_models.png"><p>DMD2、SenseFlow 使用官方已训练权重及各自 LCM 4 步采样器。它们的训练预算远大于此试验，因此这里仅用于检查效果差距。</p>
<h2>训练过程与复核</h2><img src="training_curves.png"><p>高噪声 PCM、低噪声 DMD、交接状态停止梯度；AdamW、fake 学习率 5e-7、5:1 更新比例、梯度裁剪 10。SDXL 原生 VP/DDIM 转移替代视频模型的 RF 转移。此轮使用恒定学习率，无 GAN、无 EMA。</p>
<p>实验目录：<code>/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot</code>。各组参数、源码摘要、逐步记录与四卡模型检查点均留存；参考组另已导出通用 UNet，其余组可按记录中的命令导出。<a href="assessment/result.json">完整指标与逐图记录</a> · <a href="protocol.json">实验协议与差异说明</a> · <a href="teacher_stages/stage_metrics.json">老师阶段特征统计</a></p>
<p>四个 COCO 提示词已用于人工阶段诊断；没有宣称这些是未查看的测试数据。优化器参数对照只在独立的 LAION 留出集评分。</p><ul>{sources}</ul></html>'''
(root/'report.html').write_text(body,encoding='utf-8')
print('PILOT_REPORT_RENDERED',root.resolve()/'report.html')
