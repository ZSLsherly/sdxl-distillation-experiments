import os,sys,json,csv,math
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/sdxl-crossdistill/benchmark_curves_20261018'
OUT.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
sys.path.insert(0,str(ROOT/'.plot_runtime_pkgs'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
FONT=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','savefig.facecolor':'white','svg.fonttype':'none','pdf.fonttype':42})
rows=json.loads((OUT/'summary.json').read_text())
manifest=json.loads((OUT/'manifest.json').read_text())
launch=datetime.strptime(manifest['launch']['timestamp'],'%Y%m%dT%H%M%S').replace(tzinfo=timezone(timedelta(hours=8)))
for r in rows:
    r['hours_since_launch']=(datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z')-launch).total_seconds()/3600
colors={'joint4':'#2864BD','gan4':'#D96820'}
labels={'joint4':'第4步 GAN + DMD','gan4':'第4步纯 GAN'}
fig,axs=plt.subplots(3,2,figsize=(13,11.4))
fig.subplots_adjust(top=.86,bottom=.11,left=.075,right=.97,hspace=.48,wspace=.22)
fig.suptitle('SDXL：不同训练阶段的独立评估分数',fontsize=21,x=.075,ha='left',y=.98)
fig.text(.075,.935,'固定4张留出图均值 · 每2次学生更新评估一次 · 有效 batch 64 · 原始数据连线，无平滑',fontsize=11,color='#555555')
for col,(x,xlabel) in enumerate([('student_updates','学生更新次数（G）'),('hours_since_launch','启动后经过时间（小时，含评估与保存）')]):
    for ax,metric,title,zoom in [(axs[0,col],'niqe','NIQE ↓  全程（纵轴为对数）',False),(axs[1,col],'niqe','NIQE ↓  G16起放大，便于比较后期',True),(axs[2,col],'clip_cosine','CLIP ViT-B/32 cosine ↑  文本匹配',False)]:
        for v in ['joint4','gan4']:
            rs=sorted([r for r in rows if r['variant']==v],key=lambda r:r['student_updates'])
            if zoom:rs=[r for r in rs if r['student_updates']>=16]
            ax.plot([r[x] for r in rs],[r[metric] for r in rs],color=colors[v],lw=1.6,marker='.',ms=3,label=labels[v])
            ax.scatter(rs[-1][x],rs[-1][metric],s=36,color=colors[v],zorder=4)
        base=[r[metric] for r in rows if r['student_updates']==0]
        assert max(base)-min(base)<1e-5,base
        if not zoom:ax.axhline(base[0],c='#888888',ls='--',lw=1,label='蒸馏前 SDXL 同一4步设置')
        ax.set_title(title,loc='left',fontsize=13,pad=10)
        ax.set_xlabel(xlabel);ax.grid(axis='both',alpha=.17);ax.set_xlim(left=0)
        if col==0:ax.set_xticks(np.arange(0,161,20))
        ax.tick_params(labelsize=10)
        ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        if metric=='niqe' and not zoom:
            ax.set_yscale('log');ax.set_yticks([5,10,20,50,100,300]);ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter());ax.minorticks_off()
        if zoom:ax.set_xlim(left=16 if col==0 else min(r['hours_since_launch'] for r in rows if r['student_updates']>=16))
        if metric=='clip_cosine':ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
handles,legendlabels=axs[0,0].get_legend_handles_labels()
fig.legend(handles,legendlabels,loc='upper left',bbox_to_anchor=(.067,.916),frameon=False,ncol=3,fontsize=11)
fig.text(.075,.058,'本次快照：联合组 G0–114；纯GAN组 G0–154。纯GAN保留全部已选评估点，不外推、不截短。',fontsize=10,color='#444444')
fig.text(.075,.032,'小样本趋势，非官方 COCO/FID benchmark；NIQE 不能单独代表纹理细节质量。两组前3步方案相同。',fontsize=10,color='#666666')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'benchmark_curves.{ext}',dpi=180)
plt.close(fig)
with (OUT/'checkpoint_means.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
selected=[r for r in rows if r['student_updates'] in [0,32,64,96,114,128,154]]
table=''.join(f"<tr><td>{labels[r['variant']]}</td><td>{r['student_updates']}</td><td>{r['student_samples']:,}</td><td>{r['hours_since_launch']:.2f}</td><td>{r['niqe']:.3f}</td><td>{r['clip_cosine']:.4f}</td></tr>" for r in sorted(selected,key=lambda r:(r['student_updates'],r['variant'])))
html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SDXL 训练阶段评估曲线</title><style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;color:#223047;line-height:1.7}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}.note{{background:#f3f6fa;padding:16px;border-radius:8px}}a{{color:#2864bd}}</style><h1>SDXL：不同训练阶段的独立评估分数</h1><p>本图使用本次 10,000 张训练数据实验，两组模型均从原始 SDXL Base 初始化。第1步 TM/PCM、第2/3步 DMD，两组仅第4步的 GAN 与 GAN+DMD 不同。有效 batch 64，学习率 1e-6。</p><img src="benchmark_curves.png" alt="NIQE和CLIP随学生更新次数及耗时变化"><p><a href="benchmark_curves.pdf">PDF</a> · <a href="benchmark_curves.svg">SVG</a> · <a href="checkpoint_means.csv">所有评估节点均值 CSV</a> · <a href="per_image.csv">逐图分数 CSV</a> · <a href="manifest.json">来源及评估清单</a></p><h2>指定训练节点</h2><table><tr><th>方案</th><th>学生更新</th><th>累计学生样本</th><th>启动后小时</th><th>NIQE ↓</th><th>CLIP cosine ↑</th></tr>{table}</table><h2>口径</h2><p class="note">每个节点使用相同的4条留出提示词、每条1个固定种子。表中为4张图均值，原始 CSV 保留逐图数据和样本标准差。544次图像评估并不等于544个独立测试样本。本结果是小样本趋势，不能作为官方 COCO benchmark、FID 或显著优劣的结论。图中基线是蒸馏前 SDXL 的同一4步推理设置，并非标准50步 SDXL。</p><p>NIQE 使用 <a href="https://github.com/XPixelGroup/BasicSR/blob/{manifest['basicsr_commit'].strip()}/basicsr/metrics/niqe.py">BasicSR 实现</a>、官方自然图像统计参数、完整图像的 Y 通道、96像素统计块与默认两尺度；无额外缩放，边缘不足完整统计块的部分按算法裁去。官方 baboon 样例复算值为 {manifest['niqe_sanity']:.7f}。数值越低表示更符合其自然图像统计，但不能单独判断纹理、结构或提示词是否正确。</p><p>文本匹配使用 <a href="https://github.com/openai/CLIP">OpenAI CLIP</a> ViT-B/32、FP32、标准224像素中心裁剪，记录原始余弦相似度，未乘100。不可直接与使用其他骨干、分辨率或评分缩放的论文数值比较。</p><p>横轴耗时依据实际评估目录的北京时间，减去启动时刻 2026-10-17 23:23:17，包含加载、训练、评估和保存。两组均使用2张 A100；本次快照分别截止联合组G114和纯GAN组G154。两组共同节点可按相同样本量比较，右列可按相同总耗时比较。没有插值补点、平滑或截齐较快组。</p></html>'''
(OUT/'report.html').write_text(html,encoding='utf-8')
print(json.dumps(selected,ensure_ascii=False,indent=2))
print('PLOT',OUT/'benchmark_curves.png')
