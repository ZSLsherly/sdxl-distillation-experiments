from pathlib import Path
import json,math,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path('output/sdxl-crossdistill/dmd2_matched_20261119/coco10k_official_20261121');d=r/'results'
c=json.loads((d/'complete.json').read_text());scores=c['scores'];assert scores['n']==10000 and scores['CLIP_n']==1024
assert all(math.isfinite(scores[k]) for k in ['FID','Patch_FID','CLIP'])
a=json.loads((r/'inputs/A_baseline.json').read_text());clip=json.loads((d/'clip_scores.json').read_text());assert clip['cases']==a['CLIP_cases'] and clip['revision']==a['CLIP_revision']
assert clip['n']==len(clip['alignment'])==1024
v=np.asarray(clip['alignment']);old=np.asarray(a['CLIP_per_case']['alignment']);delta=v-old
rng=np.random.default_rng(20261121);boot=np.concatenate([delta[rng.integers(0,1024,size=(500,1024))].mean(1) for _ in range(20)]);ci=np.percentile(boot,[2.5,97.5]).tolist()
comparison={'A':{k:a[k] for k in ['FID','Patch_FID','CLIP']},'DMD2':{k:scores[k] for k in ['FID','Patch_FID','CLIP']},'DMD2_minus_A':{k:scores[k]-a[k] for k in ['FID','Patch_FID','CLIP']},'CLIP_paired_prompt_bootstrap_CI95':ci,'CLIP_better_cases':int((delta>0).sum()),'FID_n':10000,'CLIP_n':1024}
(r/'对比结果.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')
fig,axes=plt.subplots(1,3,figsize=(12,4),dpi=140)
for ax,key in zip(axes,['FID','Patch_FID','CLIP']):
 vals=[a[key],scores[key]];bars=ax.bar(['A','DMD2'],vals,color=['#486f9c','#cc7a39']);ax.set_title(key+(' (lower is better)' if key!='CLIP' else ' (higher is better)'))
 for b,vv in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,vv,f'{vv:.4f}',ha='center',va='bottom')
 ax.set_ylim(0,max(vals)*1.15);ax.grid(axis='y',alpha=.15)
fig.tight_layout();fig.savefig(r/'comparison.png');plt.close(fig)
rows=''.join(f'<tr><td>{key}</td><td>{10000 if key!="CLIP" else 1024}</td><td>{a[key]:.6f}</td><td>{scores[key]:.6f}</td><td>{scores[key]-a[key]:+.6f}</td></tr>' for key in ['FID','Patch_FID','CLIP'])
caption=json.loads((d/'manifest.json').read_text())['samples']
gallery=''.join(f'<figure><a href="results/images/{p.name}"><img src="results/images/{p.name}"></a><figcaption>#{int(p.stem)} · {html.escape(caption[int(p.stem)]["prompt"])}</figcaption></figure>' for p in sorted((d/'images').glob('*.png')))
body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DMD2 与 A：COCO10K 完整对比</title><style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 22px;color:#1b2638;background:#f7f8fb}}section{{background:white;padding:24px;margin:20px 0;border-radius:10px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%}}.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}}figure{{margin:0}}.note{{background:#fff2d1;padding:14px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style><h1>DMD2 自训 G160 与 A：完整评测</h1><p>2026-11-21 · COCO2014 固定10K · 官方随机再加噪采样 · CLIP历史固定1024</p><section><h2>完整分数</h2><table><tr><th>指标</th><th>样本数</th><th>A</th><th>DMD2</th><th>DMD2−A</th></tr>{rows}</table><img src="comparison.png"><p>FID、Patch FID越低越好，CLIP越高越好。CLIP逐提示词差值95% bootstrap区间为[{ci[0]:+.4f}, {ci[1]:+.4f}]，1024条中{int((delta>0).sum())}条提高；该区间只反映这批提示词的抽样差异，不覆盖不同训练种子。</p></section><section><h2>可比条件与差异</h2><p>两组相同10K训练子集、原始SDXL起点、学生G160、batch64、CFG8及公共学习率；A为PCM1＋随机中段边界DM＋末步DM，无先验无GAN无1%混合，fake320次。DMD2为clean DM＋原生共享fake判别头，fake800次。两者不是等fake更新量或等计算预算。</p><p>评分输入是同一10000条COCO提示词/真实图/顺序，初始种子20261020+原索引，分辨率1024。A用自身DDIM4；DMD2用官方x0→独立再加噪四步999/749/499/249，CFG1。DMD2的额外噪声使用固定逐样本、逐阶段种子，与历史公开DMD2评测一致，换分卡和批量不改变随机流。</p><p>FID：中心方形→LANCZOS512→clean Inception299；Patch FID：原生中心299后clean Inception299。真实特征原样复用。CLIP：同一1024条，OpenAI ViT-B/32 revision {a['CLIP_revision']}，FP32，cosine×100。</p><p class="note">之前训练预览中的彩色伪影来自不匹配的DDIM采样；本次全部使用已验证的官方随机再加噪方式，未用错误预览计算分数。两条预检图与官方函数逐像素相同，CLIP历史8图复算最大误差5.72e-6。</p></section><section><h2>固定样图</h2><p>按历史128张样图清单取前5张展示，未按分数挑选。</p><div class="grid">{gallery}</div></section><section><h2>证据</h2><p><a href="results/complete.json">完整完成记录</a> · <a href="results/scores.json">FID原始结果</a> · <a href="results/clip_scores.json">CLIP逐样本结果</a> · <a href="results/manifest.json">评测清单和权重SHA256</a> · <a href="对比结果.json">汇总差值</a></p><p>服务器：172.18.132.41。权重与全部特征留在服务器；本地仅取回评分、记录及少量固定图片。rclone取回，逐文件SHA256核验。</p></section></html>'''
(r/'report.html').write_text(body,encoding='utf-8')
(r/'完整评测结果.txt').write_text('DMD2官方采样COCO10K/CLIP1024完整结果，2026-11-21\n'+json.dumps(comparison,ensure_ascii=False,indent=2)+'\n来源：results/complete.json、scores.json、clip_scores.json；A历史同协议结果见inputs/A_baseline.json。DMD2 F800，A F320，非等计算预算。\n',encoding='utf-8')
print(json.dumps(comparison,ensure_ascii=False))
