import os,sys,json,csv
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/sdxl-crossdistill/benchmark_curves_three_20261018'
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
variants=['joint4','gan4','joint34']
colors={'joint4':'#2864BD','gan4':'#D96820','joint34':'#218467'}
labels={'joint4':'第4步 GAN+DMD · 2卡','gan4':'第4步纯 GAN · 2卡','joint34':'第3、4步 GAN+DMD · 4卡'}
for r in rows:
    cfg=manifest['runs'][r['variant']]
    launch=datetime.strptime(cfg['launch_timestamp'],'%Y%m%dT%H%M%S').replace(tzinfo=timezone(timedelta(hours=8)))
    r['hours_since_launch']=(datetime.strptime(r['timestamp'],'%Y%m%dT%H%M%S%z')-launch).total_seconds()/3600
    r['gpu_hours']=r['hours_since_launch']*cfg['world_size']
last={v:max(r['student_updates'] for r in rows if r['variant']==v) for v in variants}
common=max(set.intersection(*[{r['student_updates'] for r in rows if r['variant']==v} for v in variants]))
baseline={m:next(r[m] for r in rows if r['variant']=='joint4' and r['student_updates']==0) for m in ['niqe','clip_cosine']}

def panel(ax,metric,x,title,zoom=False):
    for v in variants:
        rs=sorted([r for r in rows if r['variant']==v and (not zoom or r['student_updates']>=16)],key=lambda r:r['student_updates'])
        ax.plot([r[x] for r in rs],[r[metric] for r in rs],c=colors[v],lw=1.7,marker='.',ms=3,label=labels[v])
        ax.scatter(rs[-1][x],rs[-1][metric],s=34,c=colors[v],zorder=4)
    if not zoom:ax.axhline(baseline[metric],c='#888888',ls='--',lw=1,label='蒸馏前 SDXL（旧联合组）')
    ax.set_title(title,loc='left',fontsize=13,pad=10);ax.grid(alpha=.17)
    if x=='student_updates':
        ax.set_xlabel('学生更新次数（G）');ax.set_xticks(np.arange(0,161,20));ax.set_xlim(left=16 if zoom else 0)
    else:
        ax.set_xlabel('启动后经过时间（小时，含评估、保存）' if x=='hours_since_launch' else 'GPU 小时（卡数 × 启动后小时）')
        ax.set_xlim(left=0)
    ax.tick_params(labelsize=10);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    if metric=='niqe' and not zoom:
        ax.set_yscale('log');ax.set_yticks([5,10,20,50,100,300]);ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter());ax.minorticks_off()
    if metric=='clip_cosine':ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))

fig,axs=plt.subplots(3,2,figsize=(13.5,11.6))
fig.subplots_adjust(top=.845,bottom=.14,left=.073,right=.97,hspace=.48,wspace=.22)
fig.suptitle('SDXL：三组实验的独立评估曲线',fontsize=21,x=.073,ha='left',y=.98)
fig.text(.073,.938,'同4条留出提示词 · 每条固定种子102 · batch 64 · 每2次学生更新评估 · 原始均值，无平滑',fontsize=11,color='#555555')
for col,x in enumerate(['student_updates','hours_since_launch']):
    panel(axs[0,col],'niqe',x,'NIQE ↓  全程（纵轴为对数）')
    panel(axs[1,col],'niqe',x,'NIQE ↓  G16起放大',True)
    panel(axs[2,col],'clip_cosine',x,'CLIP ViT-B/32 cosine ↑  文本匹配')
handles,ll=axs[0,0].get_legend_handles_labels()
fig.legend(handles,ll,loc='upper left',bbox_to_anchor=(.065,.921),frameon=False,ncol=2,fontsize=11)
fig.text(.073,.074,f'快照截止：第4步联合 G{last["joint4"]}；第4步纯GAN G{last["gan4"]}；第3、4步联合 G{last["joint34"]}。各组曲线保留全部进度。',fontsize=10,color='#444444')
fig.text(.073,.047,'时间曲线使用不同卡数：绿色4卡，其余2卡；速度差异包含资源与配置变化。按算力对照见报告。',fontsize=10,color='#666666')
fig.text(.073,.023,'仅4个固定留出样本的趋势，非官方 COCO/FID benchmark；NIQE 不能单独代表纹理质量。',fontsize=10,color='#666666')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'benchmark_curves.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(13.5,4.8))
fig.subplots_adjust(top=.72,bottom=.18,left=.073,right=.97,wspace=.22)
panel(axs[0],'niqe','gpu_hours','NIQE ↓  G16起放大',True)
panel(axs[1],'clip_cosine','gpu_hours','CLIP cosine ↑')
fig.suptitle('按 GPU 小时比较（包含加载、评估与保存）',fontsize=18,x=.073,ha='left',y=.98)
fig.legend(handles[:3],ll[:3],loc='upper left',bbox_to_anchor=(.065,.88),frameon=False,ncol=3,fontsize=10)
fig.savefig(OUT/'gpu_hours.png',dpi=180);plt.close(fig)
with (OUT/'checkpoint_means.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
matched=[r for r in rows if r['student_updates']==common]
latest=[r for r in rows if r['student_updates']==last[r['variant']]]
def table(rs):
    head='<table><tr><th>方案</th><th>G</th><th>耗时/h</th><th>GPU小时</th><th>NIQE ↓</th><th>CLIP cosine ↑</th></tr>'
    return head+''.join(f'<tr><td>{labels[r["variant"]]}</td><td>{r["student_updates"]}</td><td>{r["hours_since_launch"]:.2f}</td><td>{r["gpu_hours"]:.2f}</td><td>{r["niqe"]:.4f}</td><td>{r["clip_cosine"]:.4f}</td></tr>' for r in rs)+'</table>'
with (OUT/'per_image.csv').open(encoding='utf-8') as f:per_image=list(csv.DictReader(f))
paired=[]
for v in ['joint4','gan4']:
    for metric in ['niqe','clip_cosine']:
        a={r['original_index']:float(r[metric]) for r in per_image if r['variant']=='joint34' and int(r['student_updates'])==common}
        b={r['original_index']:float(r[metric]) for r in per_image if r['variant']==v and int(r['student_updates'])==common}
        ds=[a[k]-b[k] for k in a]
        paired.append(dict(against=v,metric=metric,mean_delta=float(np.mean(ds)),per_prompt_delta=dict(zip(a,ds)),better_count=sum(d<0 if metric=='niqe' else d>0 for d in ds)))
summary=dict(common_step=common,matched=matched,latest=latest,paired_deltas=paired)
(OUT/'comparison_summary.json').write_text(json.dumps(summary,indent=2))
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SDXL 三组 benchmark 曲线</title><style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1350px;margin:36px auto;padding:0 24px;line-height:1.7;color:#223047}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}.note{{background:#eef3f8;padding:16px}}a{{color:#2864bd}}</style><h1>SDXL 三组 benchmark 曲线</h1><p>本次数据快照：{manifest['created_at']}。三组使用相同10,000张训练集、相同留出提示词和种子102，batch64、学生/fake学习率1e-6。第3、4步联合组在第3步新增GAN，保留原有DMD；旧两组分别只在第4步使用GAN+DMD或纯GAN。</p><img src="benchmark_curves.png"><p><a href="benchmark_curves.pdf">PDF</a> · <a href="benchmark_curves.svg">SVG</a> · <a href="checkpoint_means.csv">全部节点均值CSV</a> · <a href="per_image.csv">逐图分数CSV</a> · <a href="manifest.json">评估来源</a></p><h2>同样G{common}的比较</h2>{table(matched)}<h2>各组最新节点（步数不同）</h2>{table(latest)}<h2>按资源消耗比较</h2><img src="gpu_hours.png"><p class="note">旧两组各使用2张A100，新组使用4张A100，并把学生优化器从CPU移到GPU。因此墙钟时间优势不等同于方法优势。GPU小时为实际卡数×从各自启动到评估目录时间的小时数，包含加载、评估与保存，不是硬件性能计数器测量的计算量。</p><h2>评估口径</h2><p>沿用原图全部NIQE、CLIP设置。NIQE使用<a href="https://github.com/XPixelGroup/BasicSR/blob/{manifest['basicsr_commit'].strip()}/basicsr/metrics/niqe.py">BasicSR实现</a>和官方统计参数，对1024原图Y通道按标准96像素统计块计算；默认两尺度、无额外缩放。CLIP使用<a href="https://github.com/openai/CLIP">OpenAI CLIP</a> ViT-B/32、FP32、标准224中心裁剪和原始文本余弦相似度，未乘100。旧分数按图片SHA256及提示词复用，新增图片使用同一实现补算。</p><p class="note">只有4条固定留出提示词、每条1个种子；多个检查点不是新增独立测试样本。这些分数用于观察趋势，不能宣称官方COCO/FID成绩或统计显著优势。NIQE只衡量与自然图像统计的接近程度，无法单独证明纹理细节更好。图中虚线为旧联合组蒸馏前的同一四步设置；各组G0实测点也各自保留。</p></html>'''
(OUT/'report.html').write_text(page,encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('PLOT',OUT/'benchmark_curves.png')
