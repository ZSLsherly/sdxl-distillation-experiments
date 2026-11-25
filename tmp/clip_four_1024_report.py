import json,html,sys
from pathlib import Path
p=Path('output/sdxl-crossdistill/semantic_four_1024_20261103')
d=json.loads((p/'results.json').read_text(encoding='utf8')); n=d['n'];assert n==1024
arms=['A','B','C','C_mix1']
labels={'A':'A · 无先验','B':'B · 高噪声','C':'C · 高低噪声','C_mix1':'C + 1% 混合'}
def table(head,rows):
    return '<table><thead><tr>'+''.join('<th>'+k+'</th>' for k in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'
means=[];attrs=[]
for a in arms:
    v=d['metrics'][a];z=d['per_case'][a];assert len(z['alignment'])==len(z['rank'])==1024
    means.append([labels[a],f"{v['cosine_x100']:.3f}",f"{v['top1_correct']}/1024 ({100*v['top1_correct']/n:.1f}%)",f"{v['top5_correct']}/1024 ({100*v['top5_correct']/n:.1f}%)",f"{sum(z['alignment'][128:])/896:.3f}"])
for label,cat in [('数量 two / three','count'),('颜色','color'),('姿态 sitting / standing','pose')]:
    v=d['metrics']['A'][cat]; attrs.append([label,v['n'],*[d['metrics'][a][cat]['original_preferred'] for a in arms]])
def fmt(v):return f"{v['mean']:+.3f} [{v['paired_ci95'][0]:+.3f}, {v['paired_ci95'][1]:+.3f}]"
pairs=[[k,fmt(v),f"{v['better_count']}/1024",fmt(d['additional896_differences'][k])] for k,v in d['differences'].items()]
doc="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>四组 G160 · COCO 1024 语义对照</title><style>body{max-width:940px;margin:34px auto;padding:0 15px;font:16px/1.65 system-ui;color:#223648}table{border-collapse:collapse}td,th{padding:5px 15px;border-bottom:1px solid #ddd;text-align:left}</style><h1>四组 G160 · COCO 1,024 条配对语义评测</h1><p>在同一组 COCO2014 固定 10K 评测清单上，保留最初的 128 条，从剩余条目随机、无放回抽取 896 条（种子 20261103）。四组共 4,096 个图文对。逐文件验证 G160 权重 SHA-256；同一条目使用相同提示、种子与四步 SDXL 推理协议。原 128 张图复用原始 PNG，其余沿用原采样器和整图 VAE 解码后直接编码。</p>"""
doc+='<h2>图文对齐和找回原句</h2>'+table(['模型','CLIP 余弦 ×100 ↑','Top-1 ↑','Top-5 ↑','只看新增 896 条的余弦 ×100'],means)
doc+='<p>原始 caption 和整图分别用 OpenAI CLIP ViT-B/32 编码。文本检索候选库为全部 1,024 条 caption，完全相同的句子视为等价。候选库大于上一轮的 128 条，不宜跨轮比较 Top-1 的绝对值。</p>'
doc+='<h2>同提示词、同种子的分数差</h2>'+table(['前者减后者','均值差 [95% 区间]','前者更高','仅新增样本 [95% 区间]'],pairs)
doc+='<p>2 万次按提示词配对重采样。区间未作多重比较调整，只反映这批提示词的抽样不确定性，不涵盖独立训练种子变动。</p>'
doc+='<h2>换词后仍更偏好原句的次数</h2>'+table(['替换词','有效样本','A','B','C','C + 1%'],attrs)
doc+='<p>按数量、颜色、姿态顺序，每条 caption 最多用一种替换。这是 CLIP 对两个句子的相对偏好，不能代替人工核验的属性正确率。</p><p><a href="results.json">逐样本分数、索引、种子、排名与替换句</a></p></html>'
(p/'report.html').write_text(doc,encoding='utf8')
print(means,pairs,attrs,sep='\n')
