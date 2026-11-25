"""Render preselected examples and descriptive six-model FID comparisons."""
from pathlib import Path
from html import escape
import json,textwrap
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path('/data5/shenglin.zeng/sdxl_crossdistill')
OUT=ROOT/'outputs/coco10k_g160_20261020'
m=json.loads((OUT/'manifest.json').read_text())
scores=json.loads((OUT/'scores.json').read_text())
methods=['sdxl','dmd2','dmd_only','gan4','joint4','joint34']
labels={'sdxl':'SDXL / 50 steps','dmd2':'DMD2 / 4 steps','dmd_only':'TM + DMD / G160',
        'gan4':'GAN at step 4 / G160','joint4':'GAN+DMD at 4 / G160','joint34':'GAN+DMD at 3,4 / G160'}
font_candidates=list(Path('/usr/share/fonts').rglob('DejaVuSans.ttf'))
font=lambda size: ImageFont.truetype(str(font_candidates[0]),size) if font_candidates else ImageFont.load_default(size=size)
selected=[m['visual_indices'][i] for i in np.linspace(0,127,24,dtype=int)]
(OUT/'panels').mkdir(exist_ok=True)
panels=[]
for page in range(4):
    ids=selected[page*6:page*6+6]
    for patch in [False,True]:
        cell=299 if patch else 280
        w=230+6*(cell+10);h=90+6*(cell+20)
        im=Image.new('RGB',(w,h),'#f5f7fa');d=ImageDraw.Draw(im)
        d.text((12,12),'Native center 299px crops' if patch else 'Full images, resized only for display',font=font(23),fill='#24374a')
        for j,method in enumerate(methods):
            d.text((230+j*(cell+10),52),labels[method],font=font(14),fill='#24374a')
        for row,idx in enumerate(ids):
            y=90+row*(cell+20)
            txt=f"COCO #{idx}\nSeed {m['samples'][idx]['seed']}\n\n"+'\n'.join(textwrap.wrap(m['samples'][idx]['prompt'],27))
            d.multiline_text((12,y+10),txt,font=font(15),fill='#24374a',spacing=5)
            for j,method in enumerate(methods):
                src=Image.open(OUT/'images'/method/f'{idx:05d}.png').convert('RGB')
                if patch:
                    # Match torchvision center_crop rounding exactly.
                    x=int(round((src.width-299)/2));yy=int(round((src.height-299)/2))
                    src=src.crop((x,yy,x+299,yy+299))
                else:src=src.resize((cell,cell),Image.Resampling.LANCZOS)
                im.paste(src,(230+j*(cell+10),y))
        name=f'page{page+1}_{"texture" if patch else "full"}.png'
        im.save(OUT/'panels'/name)
        panels.append(name)
rows=[]
for method in methods:
    s=scores[method]
    delta='' if method in ('sdxl','dmd2','dmd_only') else f'{s["FID"]-scores["dmd_only"]["FID"]:+.3f} / {s["Patch_FID"]-scores["dmd_only"]["Patch_FID"]:+.3f}'
    rows.append(f'<tr><td>{escape(labels[method])}</td><td>{s["n"]}</td><td>{s["FID"]:.4f}</td><td>{s["Patch_FID"]:.4f}</td><td>{delta}</td></tr>')
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>COCO2014 10K 六模型完整评测</title><style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1500px;margin:30px auto;padding:0 20px;line-height:1.8;color:#203247;background:#f6f8fa}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #dce4ed;text-align:left}th{background:#e7eef4}img{width:100%;height:auto}a{color:#1769a1}.note{padding:16px;background:#e8f2f8}h2{margin-top:30px}</style>
<h1>COCO-2014 固定10K：四组G160与原始SDXL、DMD2</h1>
<p>每个模型完整生成10,000张1024×1024图片；本表为本次实测，不是论文引用数值。G160表示学生优化器更新160次。</p>
<table><tr><th>模型</th><th>样本</th><th>FID ↓</th><th>Patch FID ↓</th><th>相对纯DMD：ΔFID / ΔPatch FID</th></tr>'''+''.join(rows)+'''</table>
<div class="note"><b>解释：</b>负差值表示该指标改善。FID反映整体分布，Patch FID反映中心局部分布；低Patch FID不能单独证明所有纹理都更真实。各方案只有一次训练运行，本次是固定测试集上的描述性比较，不是跨训练种子的显著性结论。原始SDXL50步为质量参照，DMD2为官方充分训练四步参照。</div>
<h2>统一评测协议</h2><ul>
<li>官方DMD2 COCO10K CSV及对应真实图片；每题一个固定种子，六模型共享初始噪声。</li>
<li>FID：真实图中心裁方后缩至512，生成图缩至512；clean-FID后续标准299特征预处理。</li>
<li>Patch FID：原生分辨率中心299×299；短边不足299的真实图按官方实现不裁剪，再进入标准特征预处理。</li>
<li>自训模型保留训练对应999/919/499/199时间步；DMD2保留官方研究版x0预测/重新加噪四步；原始SDXL默认调度器50步、CFG6。</li>
<li>统一BF16推理网络、FP32 VAE和Inception；关闭水印；不对展示图锐化或调色。</li>
<li>评分保留全部特征；128个可视化样本在运行前固定。下面24个样本从这128个等间隔取出，未按结果挑选。</li>
</ul><h2>整图与原生局部</h2>'''
for name in panels:html+=f'<p><a href="panels/{name}">打开原尺寸</a></p><img loading="lazy" src="panels/{name}" alt="{name}">'
html+='<p>完整配置、权重SHA256、样本种子、预检结果保存在 manifest.json、validation.json。评分原始文件为 scores.json，全部特征位于 features/。</p></html>'
(OUT/'report.html').write_text(html,encoding='utf-8')
print('REPORT_READY',OUT/'report.html')
