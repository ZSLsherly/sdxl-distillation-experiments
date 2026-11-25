"""Report complete 10K scores while retaining explicit status for unfinished controls."""
from pathlib import Path
from html import escape
from datetime import datetime
from zoneinfo import ZoneInfo
import json, csv, textwrap, zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt=None

ROOT=Path('/data5/shenglin.zeng/sdxl_crossdistill')
BASE=ROOT/'outputs/coco10k_g160_20261020'
OUT=BASE/'snapshot_20261021'
OUT.mkdir(exist_ok=True)
m=json.loads((BASE/'manifest.json').read_text())
scores=json.loads((BASE/'scores_ready.json').read_text())
if (BASE/'scores.json').exists(): scores.update(json.loads((BASE/'scores.json').read_text()))
methods=['sdxl','dmd2','dmd_only','gan4','joint4','joint34']
labels={'sdxl':'SDXL 50-step','dmd2':'Official DMD2 4-step','dmd_only':'TM + DMD','gan4':'Step 4: GAN','joint4':'Step 4: GAN+DMD','joint34':'Steps 3,4: GAN+DMD'}
zh={'sdxl':'原始 SDXL 50 步','dmd2':'官方 DMD2 4 步','dmd_only':'纯 DMD（保留第1步 TM）','gan4':'第4步纯 GAN','joint4':'第4步 GAN+DMD','joint34':'第3、4步 GAN+DMD'}
counts={}
for method in methods:
    counts[method]=sum(int(p.stem.split('_')[1])-int(p.stem.split('_')[0]) for p in (BASE/'features'/method).glob('*.npz'))
now=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S +08:00')
state={'snapshot_time':now,'counts':counts,'scores':scores,'selected_indices':[m['visual_indices'][i] for i in np.linspace(0,127,24,dtype=int)]}
(OUT/'results.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8')
with (OUT/'scores.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['method','generated','FID','Patch_FID','delta_FID_vs_DMD','delta_PatchFID_vs_DMD'])
    for key in methods:
        s=scores.get(key,{})
        w.writerow([key,counts[key],s.get('FID',''),s.get('Patch_FID',''),s.get('FID',0)-scores['dmd_only']['FID'] if s else '',s.get('Patch_FID',0)-scores['dmd_only']['Patch_FID'] if s else ''])

if plt is not None:
    ready=[k for k in methods if k in scores]
    colors={'sdxl':'#7c8796','dmd2':'#604fc2','dmd_only':'#4278b4','gan4':'#cd8b3b','joint4':'#2b9c8e','joint34':'#c05f8a'}
    fig,axes=plt.subplots(1,2,figsize=(13,4.8),layout='constrained')
    for ax,metric in zip(axes,['FID','Patch_FID']):
        vals=[scores[k][metric] for k in ready]
        bars=ax.barh([labels[k] for k in ready],vals,color=[colors[k] for k in ready])
        ax.invert_yaxis();ax.set_xlim(0,max(vals)*1.15);ax.set_title(metric.replace('_',' ')+' (lower is better)')
        ax.bar_label(bars,fmt='%.3f',padding=5);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('COCO-2014 fixed 10K | complete sets only | G160 student checkpoints',fontsize=13)
    fig.savefig(OUT/'scores.png',dpi=170);fig.savefig(OUT/'scores.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,metric in zip(axes,['FID','Patch_FID']):
        keys=['gan4','joint4','joint34']
        vals=[scores[k][metric]-scores['dmd_only'][metric] for k in keys]
        bars=ax.barh([labels[k] for k in keys],vals,color=['#20816b' if v<0 else '#b4574c' for v in vals])
        ax.axvline(0,color='#444',linewidth=.8);ax.invert_yaxis();ax.bar_label(bars,fmt='%+.3f',padding=5)
        span=max(abs(v) for v in vals) or 1
        ax.set_xlim(min(0,min(vals))-span*.3,max(0,max(vals))+span*.4)
        ax.set_title('Delta '+metric.replace('_',' ')+' vs TM + DMD')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Negative = improved; descriptive single-run differences, not significance tests',fontsize=12)
    fig.savefig(OUT/'deltas.png',dpi=170);fig.savefig(OUT/'deltas.pdf');plt.close(fig)
    

font_paths=list(Path('/usr/share/fonts').rglob('DejaVuSans.ttf'))
def font(n):return ImageFont.truetype(str(font_paths[0]),n) if font_paths else ImageFont.load_default()
(OUT/'panels').mkdir(exist_ok=True)
panels=[]
for page in range(4):
    ids=state['selected_indices'][page*6:page*6+6]
    for patch in [False,True]:
        cell=299 if patch else 280
        im=Image.new('RGB',(230+6*(cell+10),90+6*(cell+20)),'#f5f7fa');d=ImageDraw.Draw(im)
        d.text((12,12),'Native center 299px crops' if patch else 'Full images; display resize only',font=font(23),fill='#24374a')
        for j,key in enumerate(methods):d.text((230+j*(cell+10),52),labels[key],font=font(14),fill='#24374a')
        for row,idx in enumerate(ids):
            y=90+row*(cell+20)
            txt=f"COCO #{idx}\nSeed {m['samples'][idx]['seed']}\n\n"+'\n'.join(textwrap.wrap(m['samples'][idx]['prompt'],27))
            d.multiline_text((12,y+10),txt,font=font(15),fill='#24374a',spacing=5)
            for j,key in enumerate(methods):
                p=BASE/'images'/key/f'{idx:05d}.png'
                if not p.exists():
                    d.text((240+j*(cell+10),y+cell//2),'Pending generation',font=font(18),fill='#767676');continue
                img=Image.open(p).convert('RGB')
                if patch:
                    x=int(round((img.width-299)/2));yy=int(round((img.height-299)/2));img=img.crop((x,yy,x+299,yy+299))
                else:img=img.resize((cell,cell),Image.Resampling.LANCZOS)
                im.paste(img,(230+j*(cell+10),y))
        name=f'page{page+1}_{"texture" if patch else "full"}.png'
        im.save(OUT/'panels'/name);panels.append(name)
table=[]
for key in methods:
    s=scores.get(key)
    if s:
        delta='—' if key in ['sdxl','dmd2','dmd_only'] else f'{s["FID"]-scores["dmd_only"]["FID"]:+.3f} / {s["Patch_FID"]-scores["dmd_only"]["Patch_FID"]:+.3f}'
        cols=[zh[key],f'{s["n"]:,}/10,000',f'{s["FID"]:.3f}',f'{s["Patch_FID"]:.3f}',delta]
    else:cols=[zh[key],f'{counts[key]:,}/10,000','待完成','待完成','—']
    table.append('<tr>'+''.join('<td>'+escape(x)+'</td>' for x in cols)+'</tr>')
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>COCO10K 实验结果整理</title>
<style>body{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;line-height:1.8;color:#203247;background:#f6f8fa}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #dce4ed;text-align:left}th{background:#e7eef4}img{width:100%;height:auto}a{color:#1769a1}.note{padding:16px;background:#e8f2f8}h2{margin-top:30px}.muted{color:#6a7687}</style>
<h1>COCO-2014 固定10K：G160 四组消融与官方对照</h1>'''
html+=f'<p class="muted">结果快照：{now}。只对完整10K生成集评分；尚未完成的原始SDXL不填推测分数。</p>'
html+='<table><tr><th>模型</th><th>完成量</th><th>FID ↓</th><th>Patch FID ↓</th><th>相对纯DMD：ΔFID / ΔPatch FID</th></tr>'+''.join(table)+'</table>'
html+='<h2>指标比较</h2><img src="scores.png"><img src="deltas.png">'
html+='<div class="note"><b>结论边界：</b>负差值表示指标改善。Patch FID只反映指定中心裁剪的分布差异，不能单独证明所有纹理更真实。每种方案只有一次训练运行，本表不是显著性检验。官方DMD2训练预算和数据量与G160不同，用于观察质量差距，不属于等训练预算的算法消融。</div>'
html+='<h2>实验与评分协议</h2><ul><li>四组均从原始SDXL初始化；student更新160次，有效batch64，第一步保留TM；其余阶段按组别使用DMD/GAN。GAN只影响训练，推理都使用同一四步结构。</li><li>四组共用LAION 10K训练子集；本次COCO10K是评测数据，不用于这些训练。</li><li>自训四步时间点999/919/499/199；官方DMD2使用原研究版四步预测x0并重新加噪；原始SDXL使用50步、CFG6。</li><li>每条COCO提示词使用固定种子20261020+index。统一1024²生成、BF16网络、FP32 VAE与Inception。</li><li>FID按DMD2代码：中心裁方后缩至512，再进入clean-FID 299特征预处理。Patch FID为原生中心299²；真实图短边不足299时沿用官方不裁剪分支。</li><li>第4步纯GAN为2卡训练，其他三组为4卡，batch64一致；批次划分和数值路径不同，所以不能称逐位一致的受控实验。</li><li>本报告只比较G160终点；此前少量LAION样图的CLIP/NIQE分数不拼接进本次COCO曲线。</li></ul>'
html+='<h2>固定样本视觉对照</h2><p>128个展示样本在评测运行前固定；这里从中等间隔取24个，未按结果筛选。每页先看整图，再看原尺寸中心局部。布局发生变化时，同位置裁剪未必包含同一对象；不要把构图变化直接当作纹理改善。</p>'
for name in panels:html+=f'<p><a href="panels/{name}">查看原尺寸 {name}</a></p><img loading="lazy" src="panels/{name}">'
html+='<p><a href="results.json">原始结果与快照状态</a> · <a href="scores.csv">分数CSV</a> · <a href="scores.pdf">指标图PDF</a> · <a href="deltas.pdf">差值图PDF</a></p></html>'
(OUT/'report.html').write_text(html,encoding='utf8')
with zipfile.ZipFile(BASE/'snapshot_20261021.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if p.is_file():z.write(p,p.relative_to(OUT).as_posix())
print('SNAPSHOT_READY',OUT, 'archive_bytes', (BASE/'snapshot_20261021.zip').stat().st_size,flush=True)
