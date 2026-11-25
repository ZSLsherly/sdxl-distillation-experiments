import json,html
from pathlib import Path
p=Path('output/sdxl-crossdistill/semantic_abc_20261103'); d=json.loads((p/'results.json').read_text(encoding='utf8')); n=d['n']
def table(head,rows):
 return '<table><tr>'+''.join('<th>'+h+'</th>' for h in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
means=table(['组别','图文余弦 ×100 ↑','Top-1 找回','Top-5 找回'],[(a,round(v['cosine_x100'],3),f"{v['top1_correct']}/{n}",f"{v['top5_correct']}/{n}") for a,v in d['metrics'].items()])
diff=table(['组差','平均差','95% 配对区间','前者得分更高'],[(k,f"{v['mean']:+.3f}",' 至 '.join(f'{x:+.3f}' for x in v['paired_ci95']),f"{v['better_count']}/{n}") for k,v in d['differences'].items()])
attr=table(['替换类型','样本数','A 原句胜','B 原句胜','C 原句胜'],[(label,d['metrics']['A'][k]['n'],*[d['metrics'][a][k]['original_preferred'] for a in 'ABC']) for k,label in [('count','two ↔ three'),('color','红 ↔ 蓝 / 绿 ↔ 黄'),('pose','sitting ↔ standing')]])
page=f"""<!doctype html><html lang='zh'><meta charset='utf-8'><title>A/B/C 文图语义评测</title><style>body{{max-width:900px;margin:36px auto;padding:0 18px;font:16px/1.7 system-ui;color:#23384a}}table{{border-collapse:collapse}}th,td{{text-align:left;padding:6px 18px;border-bottom:1px solid #ccc}}</style>
<h1>A/B/C · 同提示词语义评测</h1><p>使用此前 COCO2014 10K 评测在生成前预选并实际留存的 128 个索引。A、B、C 均为原版 G160，未加 GAN 或学生权重混合。逐一核验清单、种子、完整 PNG 文件集，每个模型 128 张，共 384 张。</p>
<h2>整图与原始 caption 的对齐</h2>{means}<p>使用 OpenAI CLIP ViT-B/32，CPU FP32，官方图像预处理，原始 caption 原样编码。取归一化图像、文本嵌入的余弦乘 100。Top-k 的候选是这批 128 条 caption，分数与实验室既有其他尺度指标应分开解读。</p>
<h2>配对比较</h2>{diff}<p>按提示词索引成对重采样 20,000 次，取均值差的百分位区间。该区间描述固定模型下提示词抽样的不确定性，不涵盖训练种子波动。</p>
<h2>词语替换诊断</h2>{attr}<p>仅在原句中出现一个该类可替换词时使用，每条 caption 按数量、颜色、姿态优先级最多纳入一次。计数表示在同一张图上，CLIP 对原句打分高于改写句的次数。这是模型偏好，不是人工核验的物体计数或空间关系准确率。</p>
<p>模型公开版本 {html.escape(d['revision'])}。<a href='results.json'>逐提示词得分、种子、检索排名与改写句</a>。</p></html>"""
(p/'report.html').write_text(page,encoding='utf8')
