from pathlib import Path
import json,html,numpy as np
root=Path('output/sdxl-crossdistill');out=root/'semantic_old_uniform_20261104'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
new=read(out/'results.json');old=read(root/'semantic_four_1024_20261103/results.json')
key='joint4_uniform';assert new['revision']==old['revision'] and new['cases']==old['cases'] and len(new['cases'])==new['n']==1024
allmetrics={**old['metrics'],**new['metrics']}; own=np.array(new['per_case'][key]['alignment']); assert np.isfinite(own).all()
rng=np.random.default_rng(20261103); diffs={}; labels={key:'旧均匀四点 · 末步 GAN＋DMD','C_mix1':'C＋1% 无 GAN'}
for k in old['metrics']:
 d=own-np.array(old['per_case'][k]['alignment'])
 means=np.concatenate([d[rng.integers(0,1024,(500,1024))].mean(1) for _ in range(40)])
 diffs[k]=dict(delta=float(d.mean()),lo=float(np.percentile(means,2.5)),hi=float(np.percentile(means,97.5)))
(out/'comparison.json').write_text(json.dumps(diffs,indent=2),encoding='utf8')
def tbl(headers,rows):return '<table><tr>'+''.join('<th>'+v+'</th>' for v in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</table>'
m=new['metrics'][key]
body='<h1>旧均匀四点末步 GAN＋DMD · G160 语义评测</h1><p>实验室 172.18.132.41；COCO2014 固定 1,024 个图文对。沿用 A/B/C 同一清单、种子、文本检索池和 OpenAI CLIP ViT-B/32。复用 128 张已有图片，补生成其余 896 张。重复 caption 对检索视为等价。</p>'
body+=tbl(['实验','图文余弦 ×100','Top‑1 /1024','Top‑5 /1024','数量偏好','颜色偏好','姿态偏好'],[[labels.get(k,k),f"{v['cosine_x100']:.3f}",v['top1_correct'],v['top5_correct'],*[f"{v[c]['original_preferred']}/{v[c]['n']}" for c in ('count','color','pose')]] for k,v in allmetrics.items()])
body+=tbl(['旧均匀组减对照','平均分差','配对 95% 区间'],[[labels.get(k,k),f"{v['delta']:+.3f}",f"[{v['lo']:+.3f}, {v['hi']:+.3f}]"] for k,v in diffs.items()])
body+='<p>区间来自 20,000 次配对提示词重采样，未校正多重比较。训练种子与硬件间波动不包含在区间内。跨版本比较同时改变训练目标、采样策略和 GAN 配置。原完整 COCO10K FID 23.4191、中心 299 Patch FID 37.8199，训练用时 9.41 小时（G0 至 G160 固定图）。</p><a href="results.json">单样本分数及排名</a> · <a href="comparison.json">配对区间</a>'
(out/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>旧均匀组 COCO1024 语义</title><style>body{font:16px/1.6 system-ui;max-width:1050px;margin:40px auto;color:#233848}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}table{border-collapse:collapse;margin-bottom:25px}th{background:#edf4f5}</style>'+body+'</html>',encoding='utf8')
print(m,diffs)
