from pathlib import Path
import json,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from openpyxl import load_workbook
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement

src=Path('output/sdxl-crossdistill')
out=src/'research_brief_20261105';out.mkdir(exist_ok=True)
imgs=out/'figures';imgs.mkdir(exist_ok=True)
vis=src/'semantic_pair_1024_20261105'
sheet=load_workbook(src/'all_experiments_20261105/SDXL最新实验与消融.xlsx',read_only=True,data_only=True).worksheets[0]
data=list(sheet.values)[1:];assert len(data)==17
font='C:/Windows/Fonts/msyh.ttc'
font_manager.fontManager.addfont(font)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':13})
navy='#17364A'; teal='#087F75'; orange='#BC612B'; grey='#617480'
names=['SDXL（50步）','DMD2（4步）','旧四点 · 纯DMD','旧四点 · 末步纯GAN','旧四点 · 末步GAN+DM','旧四点 · 后两步GAN+DM','均匀四点 · 旧clean+GAN','A · 无先验','B · 高噪声先验','C · 双端先验','C+1% · 基准','C+1% · 全选DM2/3/4','C+1% · 首步换DM','C+1% · 仅末步学生DM','A · PCM×10','无先验 · 首步换DM','C+1% · 额外去噪GAN']
def figure(name,fig):
    path=imgs/(name+'.png');fig.savefig(path,dpi=190,facecolor='white');plt.close(fig);return path
def chart(ids,title,metrics,limits,filename,height):
    fig,axes=plt.subplots(1,len(metrics),figsize=(12,height),squeeze=False)
    fig.subplots_adjust(left=.248,right=.952,top=1-.86/height,bottom=.53/height,wspace=.3)
    fig.text(.03,1-.32/height,title,fontsize=18,weight='bold',color=navy)
    for c,((col,label),bounds) in enumerate(zip(metrics,limits)):
        ax=axes[0,c];ax.set_xlim(*bounds);ax.set_ylim(len(ids)-.45,-.7)
        ax.set_yticks(range(len(ids)),[names[i] for i in ids] if c==0 else ['']*len(ids))
        ax.set_title(label,loc='left',fontsize=12,color=grey,pad=8)
        ax.grid(axis='x',alpha=.15);ax.tick_params(length=0,labelsize=11);ax.spines[:].set_visible(False)
        for y,i in enumerate(ids):
            val=data[i][col];color=grey if i<2 else teal if i in (13,14,16) else navy
            ax.plot(val,y,'o',ms=6,color=color);ax.annotate(f'{val:.3f}',(val,y),xytext=(7,0),textcoords='offset points',va='center',fontsize=10,color=color)
    return figure(filename,fig)
def fid_scatter():
    fig,ax=plt.subplots(figsize=(13.1,6.75))
    fig.subplots_adjust(left=.075,right=.98,top=.85,bottom=.12)
    ax.set(xlim=(16.7,32.35),ylim=(16,57),xlabel='全图 FID（越小越好）',ylabel='中心 299×299 Patch FID（越小越好）')
    ax.set_xticks(np.arange(18,33,2));ax.set_yticks(np.arange(20,56,5))
    ax.grid(color='#E1E8EA',lw=.7);ax.set_axisbelow(True)
    ax.spines[['right','top']].set_visible(False)
    fig.text(.075,.945,'COCO10K  ·  全图与局部质量',fontsize=21,weight='bold',color=navy)
    fig.text(.075,.893,'同一模型对应一个点。左移代表全图 FID 改善，下移代表 Patch FID 改善。',fontsize=12,color=grey)
    labels=['SDXL 50步','DMD2 4步','旧四点 · 纯DM','旧四点 · 末步纯GAN','旧四点 · 末步联合','旧四点 · 后两步联合','旧 clean＋末步GAN','A 无先验','B 高噪声','C 双端先验','C＋1% 基准','全选 DM2/3/4','首步换DM（C＋1%）','仅末步学生DM','A · PCM×10','首步换DM（无先验）','额外去噪末步GAN']
    positions=[(19.05,18.6),(17.05,22.4),(24.95,54.3),(20.8,54.1),(25.05,47.4),(20.55,49.4),(20.1,39.6),(22.4,33.3),(26.9,36.2),(27.5,33.6),(27.4,31.6),(23.15,30.5),(29.9,35.8),(19.2,27.4),(22.1,36.0),(28.9,42.2),(26.1,24.5)]
    for i,(txt,xytext) in enumerate(zip(labels,positions)):
        x,y=data[i][3:5]
        color=grey if i<2 else orange if i<7 else teal if i in (13,16) else navy
        ax.scatter(x,y,s=74 if i in (13,16) else 48,marker='D' if i<2 else 'o',color=color,edgecolors='white',linewidths=.65,zorder=4)
        ax.annotate(txt,xy=(x,y),xytext=xytext,va='center',fontsize=11.5,color=color,
                    weight='bold' if i in (13,16) else 'normal',
                    arrowprops=dict(arrowstyle='-',color=color,lw=.8,alpha=.7,shrinkB=4),
                    bbox=dict(facecolor='white',edgecolor='none',pad=.4,alpha=.94))
    return figure('01_全部正式评分',fig)
overview=fid_scatter()
stage=chart([10,11,13,12,15],'学生在哪些阶段接受 DM？',[(3,'全图 FID ↓'),(4,'Patch FID ↓'),(8,'单卡 A800 小时 ↓')],[(21,34),(25,47),(5,11)],'02_阶段监督与训练耗时',4.1)
prior=chart([7,8,9,10,14],'噪声先验、fake 混合与首步 PCM 权重',[(3,'全图 FID ↓'),(4,'Patch FID ↓'),(5,'CLIP×100 ↑ · 1024 条')],[(24,27.5),(32,39),(31.42,31.84)],'03_先验与PCM',4.1)
gan=chart([2,3,4,5,10,16],'GAN 的收益更集中在局部指标',[(3,'全图 FID ↓'),(4,'Patch FID ↓')],[(23,27),(23,60)],'04_GAN监督',4.7)
res=json.loads((vis/'results.json').read_text())
fig,ax=plt.subplots(figsize=(10,2.7));fig.subplots_adjust(left=.29,bottom=.26,top=.74,right=.98)
for i,k in enumerate(['A_pcm10-A','PCM_DM4_G160-C_mix1']):
    v=res['differences'][k];x=v['mean'];lo,hi=v['paired_ci95'];ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',capsize=5,color=teal)
    ax.text(.014,i,f'{x:+.3f}   [{lo:+.3f}, {hi:+.3f}]',va='center',fontsize=11)
ax.axvline(0,color=grey,ls='--',lw=1);ax.set_xlim(-.36,.28);ax.set_ylim(1.6,-.6)
ax.set_yticks([0,1],['A×10 减 A×1','仅末步 DM 减 C+1% 基准']);ax.set_xlabel('CLIP×100 差值（负值表示图文对齐下降）',fontsize=11);ax.spines[:].set_visible(False)
fig.suptitle('同一批 COCO1024 提示词的配对差值与 95% 区间',x=.03,ha='left',color=navy,fontsize=17)
ci=figure('05_CLIP配对差值',fig)
arms=['A','A_pcm10','C_mix1','PCM_DM4_G160'];labs=['A · PCM×1','A · PCM×10','C+1% · 中段＋末步','C+1% · 仅末步']
im=Image.new('RGB',(4*300,2*346),'white');draw=ImageDraw.Draw(im);f=ImageFont.truetype(font,18)
for j,(idx,detail) in enumerate([(401,False),(193,True)]):
    for k,(a,l) in enumerate(zip(arms,labs)):
        path=vis/a/(f'{idx:05d}'+('_center' if detail else '')+'.png')
        tile=Image.open(path).convert('RGB').resize((295,295),Image.Resampling.LANCZOS)
        draw.text((k*300+3,j*346+7),l,font=f,fill=navy);im.paste(tile,(k*300,j*346+39))
qual=imgs/'06_配对整图与纹理.png';im.save(qual)

prs=Presentation();prs.slide_width=Inches(13.333);prs.slide_height=Inches(7.5)
def tb(sl,x,y,w,h,s,sz=18,color=navy,bold=False):
    tf=sl.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h)).text_frame
    tf.word_wrap=True;tf.margin_left=tf.margin_right=Inches(.02)
    for i,line in enumerate(s.split('\n')):
        p=tf.paragraphs[0] if not i else tf.add_paragraph();r=p.add_run();r.text=line
        r.font.name='Microsoft YaHei';r.font.size=Pt(sz);r.font.bold=bold;r.font.color.rgb=RGBColor.from_string(color[1:])
        ea=OxmlElement('a:ea');ea.set('typeface','Microsoft YaHei');r._r.get_or_add_rPr().append(ea)
def slide(title,sub=''):
    sl=prs.slides.add_slide(prs.slide_layouts[6]);tb(sl,.52,.3,12.3,.65,title,29,navy,True)
    if sub:tb(sl,.55,1.05,12.15,.7,sub,16,grey)
    tb(sl,.55,7.05,12,.3,f'SDXL 四步蒸馏    /    2026.11.05                                     {len(prs.slides):02d}',10,grey)
    return sl
def pic(sl,path,x,y,w,h):
    with Image.open(path) as a:r=a.width/a.height
    W=min(w,h*r);H=W/r;sl.shapes.add_picture(str(path), Inches(x+(w-W)/2), Inches(y+(h-H)/2), Inches(W), Inches(H))
def note(sl,s):tb(sl,.7,6.1,12,.84,s,17,teal,True)
sl=slide('SDXL 四步蒸馏：各阶段的监督如何影响画质？','15 个自训 G160 模型、2 个公开参照；训练集为 LAION 衍生 10K，正式分布评分采用固定 COCO2014 10K。')
for j,(a,b) in enumerate([('22.129  /  全图 FID','仅末步学生 DM 组在自训模型中最低，A800 训练 5.98 小时。'),('25.669  /  Patch FID','额外去噪末步 GAN 在自训模型中最低，全图 FID 与其基准接近。'),('−0.136  /  CLIP×100','仅末步 DM 的平均语义分比 C+1% 基准低。全图分布、局部纹理与图文对齐需要分别比较。')]):
    tb(sl,.8,2.05+j*1.43,11.6,.5,a,23,teal,True);tb(sl,.8,2.62+j*1.43,11.6,.65,b,17)
sl=slide('阶段消融：同样 160 次学生更新，哪种做法更省？','C+1% 使用双端先验，每次学生更新后将 1% 共享参数混入 fake。基准每轮随机选一个中段做 DM，再做末步 DM。')
pic(sl,stage,.4,1.75,12.5,4.1);note(sl,'保留中段 fake，只删学生中段 DM：FID −3.528，Patch FID −4.943，省约 55 分钟。等时端点因中断缺失。')
sl=slide('首步与先验：对全图有利的改动，可能损失局部细节','A 在各阶段合法区间均匀加噪；B 固定加强高噪声；C 在末步将先验配额分给高、低噪声。')
pic(sl,prior,.4,1.75,12.5,4.1);note(sl,'A 单独把 PCM 权重从 1 调到 10：FID 改善 0.976，Patch FID 恶化 1.853。B 改为 C 后，Patch FID 改善 3.048。')
sl=slide('GAN 应该替代末步 DM，还是与它一起训练？','上四行共用旧四点。下两行构成另一组配对，比较新版 C+1% 加入额外去噪 GAN 前后的结果。')
pic(sl,gan,.4,1.65,12.5,4.2);note(sl,'旧配方中，末步联合 GAN 比纯 DM 的 Patch FID 低 4.529。新版额外去噪 GAN 的 Patch FID 降 7.298，全图 FID 仅变 +0.011。')
sl=slide('FID 改善后，提示词对齐有何变化？','OpenAI CLIP ViT-B/32，图文余弦 ×100。每张图按相同提示词、相同种子配对；区间按提示词 bootstrap。')
pic(sl,ci,.4,1.8,12.55,3.5)
tb(sl,.85,5.35,11.8,1.25,'PCM×10：31.675 → 31.453。仅末步 DM：31.636 → 31.500。两组的区间均低于零；区间描述本次权重在提示词间的变动，未覆盖跨训练种子变动。',18)
sl=slide('配对图片：物体关系与局部纹理','上排整图：女子拿着三明治，面前有洋葱圈。下排中心裁剪：树枝上停着一群鸟。每一排四个模型共用提示词与种子。')
pic(sl,qual,.55,1.65,12.2,4.6);tb(sl,.65,6.35,12.1,.55,'最右列鸟羽细节更可辨；上排却把洋葱圈举到了眼前，“手拿三明治”的关系没有落实。',15,teal)
sl=slide('归因：旧 clean 配方为什么仍有优势？','旧 clean＋末步 GAN 的全图 FID 为 23.419，语义分为 31.804。A×10 分别为 24.641 和 31.453。')
for j,(a,b) in enumerate([('已验证：首步 PCM 的权重','A 的单变量对照支持 PCM×10 能改善全图 FID，但语义和局部指标下降。'),('在测：旧式局部 GAN 的贡献','A×10 保留两次 fake 更新，加末步局部解码、DINO＋RGB 判别器。计划 53 次学生 GAN、800 次判别器更新。11 月 5 日 19:24 为 G55/160。'),('仍缺：中段监督目标的独立贡献','旧组监督中段的干净图预测，新组监督带噪边界。旧组还改变 fake 更新频率与优化器，单凭两组总分无法分摊这些贡献。')]):
    tb(sl,.8,2+j*1.52,11.8,.46,a,20,teal,True);tb(sl,.8,2.53+j*1.52,11.8,.88,b,16)
sl=slide('17 个模型在同一张图上的位置')
pic(sl,overview,.2,1.02,12.93,5.99)
for lo,hi,title in [(0,7,'公开参照与实验室旧配方'),(7,17,'边界 DM 与近期消融')]:
    sl=slide('附表 · '+title,'统一 COCO10K。CLIP 只在同列、同样本规模内比较；早期留出 4 张的数值见原始总表。')
    headers=['实验','FID ↓','Patch FID ↓','CLIP 1024 ↑','CLIP 128 ↑','训练 h']
    t=sl.shapes.add_table(1+hi-lo,6, Inches(.58), Inches(1.85), Inches(12.15), Inches(4.65)).table
    for c,wd in zip(t.columns,[4.4,1.32,1.67,1.8,1.65,1.31]):c.width=Inches(wd)
    vals=[headers]+[[names[i]]+[('—' if v is None else f'{v:.3f}') for v in [data[i][3],data[i][4],data[i][5],data[i][6],data[i][8]]] for i in range(lo,hi)]
    for j,row in enumerate(vals):
        for k,s in enumerate(row):
            cell=t.cell(j,k);cell.text=s;cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string('23506A' if j==0 else ('EEF5F5' if j%2 else 'FFFFFF'))
            f=cell.text_frame.paragraphs[0].runs[0].font;f.name='Microsoft YaHei';f.size=Pt(14);f.color.rgb=RGBColor.from_string('FFFFFF' if not j else '193B4C')
    tb(sl,.65,6.62,12,.37,'耗时为训练墙钟；旧组为多卡，近期组为单卡。跨硬件的耗时不等于同卡计算开销。',12,grey)
deck=out/'SDXL四步蒸馏实验汇报_二维图版.pptx';prs.save(deck)
assert len(Presentation(deck).slides)==10
with zipfile.ZipFile(out/'实验图_PNG.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(imgs.glob('*.png')):z.write(f,f.name)
print(len(prs.slides),len(list(imgs.glob('*.png'))))
