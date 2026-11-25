"""Render a self-contained HTML results index using actual inference images."""
import html,json,statistics
from pathlib import Path

ROOT=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/gan_stage_pilot_v2')
result=json.loads((ROOT/'assessment/result.json').read_text())
trials=['dmd','gan4','joint4','joint34_matched']
names={'dmd':'A：DMD对照','gan4':'B：末步仅GAN','joint4':'C：末步DMD+GAN','joint34_matched':'D：末两步GAN（次数匹配）','initial_crossdistill':'共同短训初始化','teacher_ddim':'SDXL老师','dmd2':'官方DMD2','senseflow':'官方SenseFlow'}
metrics=result['metrics'];table=[]
for trial in trials:
    m=metrics[trial];t=m['training'];s=t['summary'];ci=m['clip_delta_vs_dmd']
    table.append(f'<tr><td>{names[trial]}</td><td>{m["clip_cosine"]:.4f}</td><td>{ci["mean"]:+.4f}<br><small>{ci["prompt_bootstrap_95pct_ci"]}</small></td><td>{m["laplacian_variance"]:.5f}</td><td>{m["seed_clip_cosine_distance"]:.4f}</td><td>{m["boundary_relative_mse"]:.5f}</td><td>{s["train_seconds"]/60:.1f}分</td><td>{t["student_step_seconds_median"]:.1f}s / {t["fake_plus_D_step_seconds_median"]:.1f}s</td><td>{s["peak_allocated_GiB"]:.2f} GiB</td></tr>')
def gallery(methods,idx,seed=101,stage=None):
    items=[]
    for method in methods:
        name=f'{method}_idx{idx}_seed{seed}'+(f'_stage{stage}' if stage else '')+'.png'
        items.append(f'<figure><figcaption>{html.escape(names[method])}</figcaption><a href="assessment/{name}"><img loading="lazy" src="assessment/{name}" alt="{html.escape(names[method])}"></a></figure>')
    return '<div class="grid">'+''.join(items)+'</div>'
examples=[]
for idx in [112,114,116,118]:
    prompt=next(r['prompt'] for r in result['records'] if r['cache_index']==idx)
    examples.append(f'<h2>留出样本 {idx} · seed101</h2><p>{html.escape(prompt)}</p>'+gallery(trials,idx))
    examples.append('<details><summary>共同初始化、老师与官方模型参照</summary>'+gallery(['initial_crossdistill','teacher_ddim','dmd2','senseflow'],idx)+'</details>')
    examples.append('<details><summary>四个节点的干净图像预测</summary>')
    for stage,t in enumerate([999,919,499,199],1):examples.append(f'<h3>节点 {stage} · 输入t={t}时的x̂₀预测</h3>'+gallery(trials,idx,stage=stage))
    examples.append('</details>')
remaining=[]
for idx in range(112,120):
    for seed in [101,102]:remaining.append(f'<h3>索引 {idx} · seed{seed}</h3>'+gallery(trials,idx,seed))
diagnostics=[]
for trial in trials:
    if trial=='dmd':continue
    d=result['discriminator_train_holdout_diagnostics'][trial]
    diagnostics.append(f'<tr><td>{names[trial]}</td><td>{d["train_real"]:.4f}</td><td>{d["heldout_real"]:.4f}</td><td>{d["heldout_fake_stage3"]:.4f}</td><td>{d["heldout_fake"]:.4f}</td></tr>')
body='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SDXL Pixel GAN 初步训练结果</title>
<style>body{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1380px;margin:32px auto;padding:0 24px;background:#fbfcfe;color:#182332}h1,h2,h3{line-height:1.4}h2{margin-top:38px}table{border-collapse:collapse;width:100%;margin:20px 0;background:white}td,th{padding:10px;vertical-align:top;text-align:left;border-bottom:1px solid #dce3ec}th{background:#edf2f8}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}figure{margin:0}figcaption{font-weight:600;margin:8px 0}img{width:100%;border-radius:6px}.note{padding:16px 20px;background:#edf4fd;border-left:4px solid #3478c8}small{color:#536477}a{color:#1763a5}details{margin:20px 0}summary{cursor:pointer}li{margin:8px 0}@media(max-width:760px){.grid{grid-template-columns:repeat(2,1fr)}table{font-size:13px}}</style></head><body>
<h1>SDXL Pixel GAN 初步训练结果</h1><p>172.18.33.60 · 4 × A100 40GB · 1024px · 999→919→499→199→clean</p>
<div class="note">四组训练完成，每组24次学生更新、120次fake score更新；A组0次判别器更新，其余组各120次；所有rank参数确实更新，初始化输出一致性与独立验证集审计通过。<b>这是少样本工程与早期趋势试验，不能据此证明GAN能或不能完全替代末步DMD。</b></div>
<p>使用从零训练的条件RGB PatchGAN；真实端是LAION的VAE重建，训练用局部近似解码，评估图为完整VAE解码。尚未训练SAM/DINO判别器。C、D组GAN学生更新均为8次，D在阶段3/4各4次；B组未单独调整纯GAN权重。这些限制必须保留。</p>
<p><a href="../reviews/gan_review_resolution_20261017.html">实施前技术方案</a> · <a href="audit.json">四卡训练审计</a> · <a href="assessment/result.json">完整指标与逐图记录</a></p>
<h2>16张独立留出图：配对诊断</h2><p>8个新prompt×2个固定seed，缓存索引112…119。ΔCLIP相对A组，区间按prompt配对重采样。Laplacian方差只表示高频变化，可能包含伪纹理与噪点；不是感知质量排名。</p><div style="overflow:auto"><table><tr><th>组别</th><th>CLIP</th><th>ΔCLIP与95%区间</th><th>Laplacian方差</th><th>跨seed特征距离</th><th>919相对MSE</th><th>训练时间</th><th>学生 / fake+D中位时间</th><th>PyTorch峰值</th></tr>'''+''.join(table)+'''</table></div>
<p>计时含同步与四卡通信；fake项包含RGB判别器更新，A组完全跳过GAN解码和判别器，下面分项计时给出实际开销。计时中位数排除了每卡前两次学生及前两次fake更新。共享服务器负载会影响各组耗时，不能将全部时间差归因于损失选择。</p>
<h2>判别器过拟合诊断</h2><p>同一中心图块协议下，取8张训练真实图、8张留出真实图、8张留出生成图。不同试验的判别器分数尺度不能直接作为图像质量比较。</p><table><tr><th>组别</th><th>训练真实logit</th><th>留出真实logit</th><th>留出阶段3 logit</th><th>留出阶段4 logit</th></tr>'''+''.join(diagnostics)+'''</table>'''+''.join(examples)+'''<details><summary>查看全部8个prompt×2个seed四组图</summary>'''+''.join(remaining)+'''</details><h2>结论范围</h2><p>本报告自动汇总已完成的训练与图像，不自动宣布最佳方法。方法判断需结合纹理是否合理、文本是否对齐、结构是否漂移，以及更多更新和随机种子的复验。没有从16张图计算FID或Patch FID。</p></body></html>'''
timing_rows=[]
for t in trials:
    m=metrics[t]['training']['timing_breakdown_median_seconds']
    timing_rows.append('<tr><td>'+names[t]+'</td>'+''.join(f'<td>{m[k]:.3f}s</td>' for k in ['rollout_seconds','d_decode_seconds','d_update_seconds','fake_score_seconds'])+'</tr>')
body=body.replace('<h2>判别器过拟合诊断</h2>','<h2>辅助更新分项计时</h2><table><tr><th>组</th><th>学生rollout</th><th>D解码</th><th>D训练</th><th>fake score训练</th></tr>'+''.join(timing_rows)+'</table><p>各项分别取中位数，不能保证中位数之和等于整体中位数。GAN学生更新还含VAE输入梯度探针，当前吞吐包括诊断成本。</p><h2>判别器过拟合诊断</h2>')
(ROOT/'report.html').write_text(body,encoding='utf-8')
(ROOT/'completion.json').write_text(json.dumps({'status':'training_and_assessment_complete','student_updates_per_trial':24,'trials':trials,'report':str(ROOT/'report.html'),'claims':'feasibility and early trends only; no converged ranking'},indent=2))
print('GAN_RESULTS_REPORT_WRITTEN',ROOT/'report.html',flush=True)
