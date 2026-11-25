from pathlib import Path
import json
import html
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
rows = json.loads((BASE/'project_audit_20261112/experiment_results.json').read_text(encoding='utf-8'))
latest = {
    'A_pcm10_mix1': (24.6290169712521, 35.31965419037607, 1794390313.8918173),
    'normalized_u_g160': (22.917469109711476, 26.66264756236194, 1794401268.177921),
    'raw_angle_g160': (23.05690331213799, 25.899364195263615, 1794412212.4620144),
}
for key, (fid, patch, completed) in latest.items():
    assert math.isfinite(fid) and math.isfinite(patch)
    rows.append(dict(key=key, FID=fid, Patch_FID=patch, n=10000,
                     family='local_prior' if key!='A_pcm10_mix1' else 'boundary',
                     source=f'172.18.32.78:/data1/shenglin.zeng/coco_missing_20261111/results/{key}/complete.json',
                     completed_unix=completed, verified_date='2026-11-13'))

descriptions = {
 'dmd_only': ('1','旧四点 · PCM＋DMD','999/919/499/199；首步PCM，后续按旧clean目标做DM，作为旧GAN消融基线。'),
 'gan4': ('2','旧四点 · 末步纯GAN','沿用旧四点，末步GAN替代该步DMD；保留前面阶段的监督。'),
 'joint4': ('3','旧四点 · 末步GAN＋DMD','沿用旧四点，末步同时接受GAN与DMD。'),
 'joint34': ('4','旧四点 · 第3/4步GAN＋DMD','沿用旧四点，GAN覆盖第三、第四步的clean预测。'),
 'joint4_uniform': ('5','均匀四点 · 旧clean＋末步GAN','999/749/499/249；PCM×10，学生DM阶段2/3/4轮转，监督clean预测；末步GAN共53次，fake与判别器各800次。'),
 'sdxl': ('R1','原始SDXL · 50步','公开基础模型，50步、CFG6；提供多步质量参照。'),
 'dmd2': ('R2','官方DMD2 · 4步','公开蒸馏权重，官方四步x₀预测/重新加噪采样、CFG1；训练预算不同。'),
 'A': ('6','A · 均匀无先验','均匀四个推理点；PCM×1＋随机第2/3步带噪边界DM＋末步DM；fake中段/末步配对两次；各合法区间均匀加噪，无GAN、无1%混合。'),
 'B': ('7','B · 高噪声先验','基于A，在每个合法DM区间把900–979分配30%抽样概率，补集70%；student与fake同分布。'),
 'C': ('8','C · 双端先验','中段同B；末步将20–100和900–979的并集合计分配30%，补集70%。'),
 'C_mix1': ('9','C＋1%学生→fake','基于C，从G1起每次学生更新后将1%学生共享参数混入fake；fake阶段embedding及Adam状态不混合。'),
 'DM234_fake3': ('10','C＋1% · DM234全选','保留PCM；每G为0.5DM2＋0.5DM3＋DM4；fake三个阶段各更新一次。相比9同时增加fake预算与中段覆盖。'),
 'DM1_mid4_fake3': ('11','C＋1% · 首步改DM','首步PCM替换为边界DM1；每G为DM1＋随机中段DM＋DM4，fake对应三个阶段更新。'),
 'PCM_DM4': ('12','C＋1% · 仅末步学生DM','保留PCM，只删除学生中段DM；fake仍保留中段一次＋末步一次。'),
 'A_pcm10': ('13','A · PCM×10','基于A，仅把PCM权重从1改为10；其余采用原A的无先验方案。'),
 'DM1_uniform': ('14','首步DM＋1% · 无先验','基于11，取消加噪先验；DM1＋随机中段DM＋DM4与三次fake、1%参数混合保留。'),
 'H20_GAN': ('15','C＋1% · 额外去噪GAN','在9基础上增加训练专用末步去噪分支：终点detach后在249加噪，学生额外预测，整图VAE＋SAM判别器；正式评测仍四步。'),
 'A10_GAN': ('16','A×10 · DINO/RGB末步GAN','在13基础上，末步clean latent局部解码接DINO＋RGB判别器；学生GAN53次、判别器800次，fake仍320次。'),
 'A_clean': ('17','A · 中段改预测干净x0','在6基础上，中段DM和fake改用该步预测的clean x0加噪；保持A的中段加噪范围，后续推理仍执行原DDIM跳步。'),
 'A_pcm10_mix1': ('18','A×10＋1%学生→fake','在13基础上，从G1起加入1%学生共享权重混入fake；无先验、无GAN。'),
 'normalized_u_g160': ('19','局部边界 · 归一化转向先验','PCM×1＋随机中段边界DM＋末步DM；加噪仅在509–739/259–489/20–239局部区间。学生用70%均匀＋30%归一化转向曲线，fake均匀并按2→3→4轮转，每G两次；无1%混合、无GAN。'),
 'raw_angle_g160': ('20','局部边界 · 原始转角先验','沿用19的局部区间、fake轮转等设置，学生先验改为未按单位距离归一化的原始转角曲线。'),
}
for row in rows:
    row['id'], row['label'], row['difference'] = descriptions[row['key']]
    row.setdefault('n',10000)
rows.sort(key=lambda r: (-2 if r['id']=='R1' else -1 if r['id']=='R2' else int(r['id'])))
assert len(rows)==22
(HERE/'results_and_recipes.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')

regular = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
bold = FontProperties(fname='C:/Windows/Fonts/msyhbd.ttc')
plt.rcParams.update({'font.family':regular.get_name(),'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig = plt.figure(figsize=(20,12),facecolor='#ffffff')
grid = fig.add_gridspec(1,2,width_ratios=[1.06,1.22],left=.055,right=.98,bottom=.18,top=.80,wspace=.10)
ax = fig.add_subplot(grid[0,0]); tab = fig.add_subplot(grid[0,1]);tab.axis('off')
colors={'public':'#7d8792','legacy':'#c47b25','boundary':'#226a9a','local_prior':'#2b8b69'}
offsets={'1':(17,8),'2':(-28,9),'3':(18,-15),'4':(-30,12),'5':(-28,8),
         '6':(-10,23),'7':(17,6),'8':(30,9),'9':(20,-19),'10':(-26,-16),
         '11':(15,6),'12':(-26,10),'13':(16,12),'14':(15,6),'15':(15,-10),
         '16':(-42,11),'17':(16,6),'18':(-35,21),'19':(-37,8),'20':(15,-14),
         'R1':(13,16),'R2':(13,-18)}
for row in rows:
    color=colors[row['family']]
    ax.scatter(row['FID'],row['Patch_FID'],s=72 if row['family']=='public' else 64,
               marker='D' if row['family']=='public' else 'o',color=color,edgecolors='white',linewidth=.7,zorder=3)
    ax.annotate(row['id'],(row['FID'],row['Patch_FID']),xytext=offsets[row['id']],textcoords='offset points',
                fontsize=11,fontweight='bold',color=color,ha='left',va='center',
                bbox=dict(boxstyle='round,pad=.12',facecolor='white',alpha=.86,edgecolor='none'),
                arrowprops=dict(arrowstyle='-',color=color,linewidth=.7),zorder=4)
ax.set(xlim=(17.4,31),ylim=(17,56),xlabel='FID ↓',ylabel='PFID / Patch FID ↓')
ax.xaxis.label.set_size(15);ax.yaxis.label.set_size(15)
ax.tick_params(labelsize=11);ax.grid(color='#dce4e9',linewidth=.7,zorder=0)
for side in ['top','right']:ax.spines[side].set_visible(False)
ax.spines['left'].set_color('#b4c1ca');ax.spines['bottom'].set_color('#b4c1ca')
ax.text(.03,.97,'越靠左下越好',transform=ax.transAxes,fontproperties=regular,fontsize=13,color='#506471',va='top')
ax.set_title('每个点对应一组完整COCO10K结果',fontproperties=regular,fontsize=15,pad=16,loc='left')

tab.text(0,1.03,'编号与实验配置',fontproperties=bold,fontsize=15)
columns=[.005,.09,.79,.995]
tab.text(columns[0],.985,'编号',fontproperties=bold,fontsize=11,va='center')
tab.text(columns[1],.985,'实验',fontproperties=bold,fontsize=11,va='center')
tab.text(columns[2],.985,'FID ↓',fontproperties=bold,fontsize=11,ha='right',va='center')
tab.text(columns[3],.985,'PFID ↓',fontproperties=bold,fontsize=11,ha='right',va='center')
tab.plot([0,1],[.966,.966],color='#b8c8d3',lw=1,transform=tab.transAxes)
for i,row in enumerate(rows):
    y=.94-i*.042
    if i%2==0:tab.add_patch(plt.Rectangle((0,y-.019),1,.041,facecolor='#f3f6f8',edgecolor='none',transform=tab.transAxes,zorder=0))
    color=colors[row['family']]
    tab.text(columns[0],y,row['id'],fontsize=11.5,fontweight='bold',color=color,va='center')
    tab.text(columns[1],y,row['label'],fontproperties=regular,fontsize=11.2,va='center',color='#233541')
    tab.text(columns[2],y,f"{row['FID']:.4f}",fontsize=11.2,ha='right',va='center',color='#233541')
    tab.text(columns[3],y,f"{row['Patch_FID']:.4f}",fontsize=11.2,ha='right',va='center',color='#233541')
tab.set_xlim(0,1);tab.set_ylim(0,1)
fig.text(.055,.945,'SDXL四步蒸馏 · FID与PFID实验总览',fontproperties=bold,fontsize=27,color='#142f42')
fig.text(.055,.904,'20组自训G160 ＋ 2个公开参照  |  固定COCO-2014 10K  |  2026-11-13核对',fontproperties=regular,fontsize=13,color='#59717f')
legend_names=[('public','公开参照'),('legacy','旧clean目标系列'),('boundary','跨区间边界DM系列'),('local_prior','旧局部区间先验系列')]
for x,(family,label) in zip([.055,.205,.405,.645],legend_names):
    fig.text(x,.863,'● '+label,fontproperties=regular,fontsize=12,color=colors[family])
fig.text(.055,.091,'PFID为原生图像中心299×299裁剪的Patch FID；两项指标均越低越好。',fontproperties=regular,fontsize=11.5,color='#536b7a')
fig.text(.055,.061,'编号19/20与A/B/C的加噪范围、fake更新方式不同；公开模型训练预算也不同，跨组差距需结合配置解释。',fontproperties=regular,fontsize=11.5,color='#536b7a')
fig.text(.055,.031,'当前“PCM×10＋末端低噪声30%先验”组尚无完整评分，因此未落点。',fontproperties=regular,fontsize=11.5,color='#536b7a')
fig.savefig(HERE/'FID_PFID_全部实验.png',dpi=180,facecolor='white')
fig.savefig(HERE/'FID_PFID_全部实验.svg',facecolor='white')
plt.close(fig)

intro='''所有自训点均为G160完整COCO10K结果。图中编号用于对应下表。1%表示每次学生更新后将1%学生共享参数混入fake。PCM是首步轨迹一致性监督，DM/DMD是分布匹配监督。'''
head='''原A（编号6）的默认方案是PCM×1＋随机第2/3步边界DM＋第4步DM，fake配对更新两次。其三个加噪支持为509–979、259–979、20–979。除旧四点组1–4外，自训推理点均为999/749/499/249。'''
body='<h1>FID / PFID实验结果与配置差异</h1><p>2026-11-13 · 最近三组补测已读取远端完整10000项完成文件。</p><img src="FID_PFID_全部实验.png" alt="22组实验FID与PFID结果图"><p>'+intro+'</p><p>'+head+'</p>'
body+='<table><tr><th>编号</th><th>实验</th><th>FID↓</th><th>PFID↓</th><th>具体差异</th></tr>'
for row in rows:
    body+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [row['id'],row['label'],f"{row['FID']:.4f}",f"{row['Patch_FID']:.4f}",row['difference']])+'</tr>'
body+='</table><p>当前结果中，自训最低FID为编号12（22.1288），最低PFID为编号15（25.6695）。19/20两组同时达到较好的两项分数，但与跨区间A/B/C不是只改一个先验变量的配对对照。18对13的变化为FID −0.0122、PFID +0.0902，未呈现清晰改善。不同训练随机种子的波动尚未被这些单次运行量化。</p>'
body+='<p>原始数值与来源路径见 results_and_recipes.json。PFID只覆盖中心裁剪的分布，图中没有包含CLIP语义得分。</p>'
css='body{font:16px/1.75 "Microsoft YaHei",sans-serif;max-width:1500px;margin:32px auto;padding:0 24px;color:#213647}h1{font-size:28px}img{width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;border-bottom:1px solid #d9e2e8;text-align:left;vertical-align:top}th{background:#edf3f7}tr:nth-child(even){background:#f7f9fa}td:nth-child(3),td:nth-child(4){white-space:nowrap}'
(HERE/'各实验差异.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FID与PFID实验结果</title><style>'+css+'</style>'+body+'</html>',encoding='utf-8')
lines=['2026-11-13 FID/PFID实验图与差异说明',intro,head,'']
for row in rows:lines.append(f"{row['id']} {row['label']} | FID {row['FID']:.4f} | PFID {row['Patch_FID']:.4f}\n{row['difference']}\n")
(HERE/'各实验差异.txt').write_text('\n'.join(lines),encoding='utf-8')
print('CREATED',len(rows),'points',HERE/'FID_PFID_全部实验.png')
