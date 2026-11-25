"""Plot all four completed G160 runs, keeping endpoint and late-window evidence separate."""
import os, sys, json, csv, statistics
from pathlib import Path
from html import escape
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/sdxl-crossdistill/benchmark_final_20261020'
os.environ['MPLCONFIGDIR']=str(OUT/'.mplconfig')
sys.path.insert(0,str(ROOT/'.plot_runtime_pkgs'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
rows=json.loads((OUT/'summary.json').read_text())
manifest=json.loads((OUT/'manifest.json').read_text())
with (OUT/'per_image.csv').open(encoding='utf-8') as f:images=list(csv.DictReader(f))
variants=['dmd_only','joint4','gan4','joint34']
labels={'dmd_only':'纯 DMD（首步 TM）','joint4':'第4步 GAN+DMD','gan4':'第4步纯 GAN','joint34':'第3、4步 GAN+DMD'}
colors={'dmd_only':'#8A45B5','joint4':'#2864BD','gan4':'#D96820','joint34':'#218467'}
data={v:sorted([r for r in rows if r['variant']==v],key=lambda r:r['student_updates']) for v in variants}
assert all(len(rs)==81 and rs[-1]['student_updates']==160 for rs in data.values())
final={v:rs[-1] for v,rs in data.items()}
window={v:{m:statistics.mean(r[m] for r in rs if 120<=r['student_updates']<=160) for m in ['niqe','clip_cosine']} for v,rs in data.items()}
details={v:{int(r['original_index']):r for r in images if r['variant']==v and int(r['student_updates'])==160} for v in variants}
effect={}
for v in variants[1:]:
 effect[v]={'niqe_delta':final[v]['niqe']-final['dmd_only']['niqe'],
            'niqe_reduction_percent':100*(final['dmd_only']['niqe']-final[v]['niqe'])/final['dmd_only']['niqe'],
            'clip_delta':final[v]['clip_cosine']-final['dmd_only']['clip_cosine'],
            'niqe_better_prompts':sum(float(details[v][i]['niqe'])<float(details['dmd_only'][i]['niqe']) for i in details[v]),
            'clip_better_prompts':sum(float(details[v][i]['clip_cosine'])>float(details['dmd_only'][i]['clip_cosine']) for i in details[v]),
            'late_niqe_delta':window[v]['niqe']-window['dmd_only']['niqe'],
            'late_clip_delta':window[v]['clip_cosine']-window['dmd_only']['clip_cosine']}
analysis={'G160':final,'G120_160_means':window,'GAN_vs_DMD':effect,
          'scope':'4 fixed holdout prompts, seed102; 21 late checkpoints reuse the same prompts, not independent samples.'}
(OUT/'analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2),encoding='utf-8')
fig,axs=plt.subplots(2,2,figsize=(13.2,9))
fig.subplots_adjust(left=.075,right=.975,top=.82,bottom=.12,hspace=.40,wspace=.22)
for i,(metric,title) in enumerate([('niqe','NIQE ↓ 自然图像统计'),('clip_cosine','CLIP cosine ↑ 文本匹配')]):
 for j,start in enumerate([0,96]):
  ax=axs[j,i]
  for v in variants:
   rs=[r for r in data[v] if r['student_updates']>=start]
   ax.plot([r['student_updates'] for r in rs],[r[metric] for r in rs],color=colors[v],lw=1.8,label=labels[v])
   ax.scatter(160,final[v][metric],color=colors[v],s=30,zorder=3)
  ax.set_title(title+(' · 全程' if j==0 else ' · 后期放大'),loc='left',fontsize=13)
  ax.set_xlabel('学生更新次数 G');ax.grid(alpha=.18);ax.set_xlim(start,163)
  if metric=='niqe' and j==0:
   ax.set_yscale('log');ax.set_yticks([5,10,20,50,100,300]);ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter());ax.minorticks_off()
  if metric=='clip_cosine':ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
fig.suptitle('SDXL 四组实验：全部完成 G160',x=.075,ha='left',y=.975,fontsize=21)
fig.text(.075,.925,'使用最新两组四卡重跑；固定4条留出提示词 × 种子102；batch64；原始均值，无平滑。',fontsize=11,color='#555')
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='upper left',bbox_to_anchor=(.065,.89),ncol=4,frameon=False,fontsize=10)
fig.text(.075,.055,'NIQE不是专门的纹理指标，CLIP不衡量纹理。仅4个样本，不能视作官方COCO/FID成绩或显著性证据。',fontsize=10,color='#555')
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'benchmark_curves.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(13.2,5.5));fig.subplots_adjust(left=.075,right=.975,top=.75,bottom=.22,wspace=.23)
short=['纯 DMD','末步 GAN+DMD','末步纯 GAN','三四步 GAN+DMD']
for ax,m,title in zip(axs,['niqe','clip_cosine'],['NIQE ↓','CLIP cosine ↑']):
 for k,v in enumerate(variants):
  ax.scatter(k,final[v][m],s=100,color=colors[v],marker='o',zorder=3)
  ax.scatter(k,window[v][m],s=90,color=colors[v],marker='D',facecolors='none',zorder=3)
  ax.plot([k,k],[final[v][m],window[v][m]],color=colors[v],alpha=.5)
  ax.annotate(f'{final[v][m]:.4f}',(k,final[v][m]),xytext=(0,-20 if final[v][m]<window[v][m] else 10),textcoords='offset points',ha='center',fontsize=10)
 ax.set_xticks(range(4),short,rotation=15);ax.set_xlim(-.5,3.5);ax.margins(y=.3);ax.grid(axis='y',alpha=.2);ax.set_title(title,loc='left')
fig.suptitle('最终节点与后期均值',x=.075,ha='left',fontsize=21,y=.97)
fig.text(.075,.87,'● G160最终值　◇ G120–160共21个节点的均值（沿用同4张图，不代表新增独立样本）',fontsize=11)
fig.savefig(OUT/'endpoint_vs_window.png',dpi=180);plt.close(fig)

def table(vals):
 s='<table><tr><th>方案</th><th>NIQE ↓</th><th>CLIP cosine ↑</th></tr>'
 for v in variants:s+=f'<tr><td>{labels[v]}</td><td>{vals[v]["niqe"]:.4f}</td><td>{vals[v]["clip_cosine"]:.4f}</td></tr>'
 return s+'</table>'
paired='<table><tr><th>方案（相对纯DMD）</th><th>NIQE变化，负为好</th><th>NIQE改善比例</th><th>NIQE更好样本数</th><th>CLIP变化</th></tr>'
for v,r in effect.items():paired+=f'<tr><td>{labels[v]}</td><td>{r["niqe_delta"]:+.4f}</td><td>{r["niqe_reduction_percent"]:+.2f}%</td><td>{r["niqe_better_prompts"]}/4</td><td>{r["clip_delta"]:+.4f}</td></tr>'
paired+='</table>'
per='<table><tr><th>样本（每格 NIQE / CLIP）</th>'+''.join(f'<th>{labels[v]}</th>' for v in variants)+'</tr>'
for idx in sorted(details['dmd_only']):per+=f'<tr><td>{idx}</td>'+''.join(f'<td>{float(details[v][idx]["niqe"]):.4f} / {float(details[v][idx]["clip_cosine"]):.4f}</td>' for v in variants)+'</tr>'
per+='</table>'
sources='<ul>'+''.join(f'<li>{labels[v]}：<code>{escape(manifest["runs"][v]["run"])}</code></li>' for v in variants)+'</ul>'
page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SDXL G160 最终对比</title>
<style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1320px;margin:32px auto;padding:0 24px;color:#233047;line-height:1.7}}img{{width:100%}}table{{width:100%;border-collapse:collapse}}th,td{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}.note{{padding:16px;background:#eef3f8}}code{{overflow-wrap:anywhere}}</style>
<h1>SDXL：四组 G160 最终结果</h1><p>数据快照 {escape(manifest['created_at'])}。第4步与第3、4步 GAN+DMD 均使用最新四卡从头重跑，不拼接旧版曲线。</p>
<div id="conclusion" class="note"><strong>判断：末步 GAN+DMD 有小幅自然度指标收益，尚未证明稳定的纹理提升；末步纯 GAN 替代 DMD 暂无优势。</strong>
<ul><li>末步联合 G160 的 NIQE 为5.7629，比纯DMD低3.23%，但只有2/4样本更好。G120–160均值5.8876对5.9356，优势收窄到约0.81%。G160的4个样本CLIP均低于纯DMD。</li>
<li>三四步联合 G160 NIQE 为5.8034，比纯DMD低2.55%，也是2/4样本更好；后期均值6.0008却高于纯DMD的5.9356，说明最终节点优势未贯穿整个后期区间。不能只凭G160认定多加第3步更优。</li>
<li>末步纯GAN的最终和后期平均NIQE、CLIP均不及纯DMD，现有证据不支持用它完全取代末步DMD。</li></ul>
<p>视觉检查：四组整体都能形成可辨认场景，但靠垫、花束、玻璃和建筑线条仍可见涂抹或细线伪影。三四步联合的餐桌与飞船构型变化较明显，这不是单纯增加细节；尚未看到覆盖四个样本的一致纹理提升。不能仅凭这些图片推断训练过拟合。</p>
<p>后续若继续验证，优先保留纯DMD与末步GAN+DMD做更大留出集、多个种子的同条件比较，并加入人工盲评。当前不据此修改或继续训练。</p></div><h2>G160评分</h2>{table(final)}{paired}
<h2>完整曲线与后期变化</h2><img src="benchmark_curves.png"><p><a href="benchmark_curves.pdf">PDF</a> · <a href="benchmark_curves.svg">SVG</a> · <a href="summary.csv">全部节点数据</a> · <a href="per_image.csv">逐图评分</a></p>
<img src="endpoint_vs_window.png"><h2>G120–160均值</h2>{table(window)}
<p>21个检查点重复使用相同4条提示词，区间均值用于观察波动，不是84个独立样本，也没有据此做显著性检验。</p>
<h2>G160整图与纹理裁剪</h2><p><a href="matched_holdout.png">整图原尺寸</a> · <a href="matched_texture.png">纹理裁剪原尺寸</a>。所有组提示词相同、种子102、1024输出。整图只缩小展示，局部384×384为原生像素。构图会随模型变化，同坐标裁剪不保证同一物体精确对应。</p><img src="matched_holdout.png"><img src="matched_texture.png">
<h2>逐样本评分</h2>{per}<h2>评价边界与来源</h2><p class="note">仅4个固定留出样本、每条1个种子；不能据此断言整体质量、泛化或过拟合，也不是官方COCO/FID评测。NIQE衡量自然图像统计偏离，无法单独证明纹理更好；CLIP反映文本匹配。四组均为160次学生更新、有效batch64、学习率1e-6，首步保留TM/PCM。末步纯GAN组为两卡，其余为四卡；设备数量、优化器驻留以及运行时GPU竞争不同，耗时不能直接用来证明算法效率。</p>
<p>NIQE沿用BasicSR实现，1024原图Y通道、96像素统计块、两尺度；CLIP沿用OpenAI ViT-B/32 FP32与标准224中心裁剪，原始余弦值。只复用同图片SHA256、提示词与样本ID的缓存分数。<a href="manifest.json">完整评估来源</a> · <a href="analysis.json">分析数据</a></p>{sources}</html>'''
(OUT/'report.html').write_text(page,encoding='utf-8')
print(json.dumps(analysis,ensure_ascii=False,indent=2))
