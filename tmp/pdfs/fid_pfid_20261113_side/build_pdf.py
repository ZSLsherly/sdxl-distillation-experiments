from pathlib import Path
import json
from xml.sax.saxutils import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, A3, landscape
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / 'output/sdxl-crossdistill/fid_pfid_comparison_20261113_side'
OUT = ROOT / 'output/pdf/SDXL_FID_PFID实验结果与配置_20261113.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
rows = json.loads((SRC/'results_and_recipes.json').read_text(encoding='utf-8'))
pdfmetrics.registerFont(TTFont('YaHei', 'C:/Windows/Fonts/msyh.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('YaHeiB', 'C:/Windows/Fonts/msyhbd.ttc', subfontIndex=0))
INK = colors.HexColor('#182b42')
MUTED = colors.HexColor('#556577')
BLUE = colors.HexColor('#1969a3')
STYLE = ParagraphStyle('body', fontName='YaHei', fontSize=9.2, leading=14.5, textColor=INK, wordWrap='CJK')
SMALL = ParagraphStyle('small', parent=STYLE, fontSize=8, leading=12, textColor=MUTED)
HEAD = ParagraphStyle('head', parent=STYLE, fontName='YaHeiB', fontSize=9.5)
c = canvas.Canvas(str(OUT), pagesize=landscape(A3), pageCompression=1)
c.setTitle('SDXL四步蒸馏：FID、PFID实验结果与配置')
c.setAuthor('实验结果整理')

def p(text, style=STYLE):
    return Paragraph(escape(str(text)).replace('x₀','x0'), style)

def para(text, y, x=42, width=None, style=STYLE):
    width = width or A4[0]-84
    q=p(text,style); _,h=q.wrap(width,1000); q.drawOn(c,x,y-h)
    return y-h-9

def footer(n, w):
    c.setStrokeColor(colors.HexColor('#dce3e9')); c.line(42,34,w-42,34)
    c.setFont('YaHei',8); c.setFillColor(MUTED)
    c.drawString(42,21,'COCO-2014 固定10K | 核验日期：2026-11-13')
    c.drawRightString(w-42,21,f'{n} / 4')

def start_page(n,title,subtitle):
    c.setPageSize(A4)
    w,h=A4
    c.setFillColor(BLUE); c.rect(42,h-46,34,4,fill=1,stroke=0)
    c.setFillColor(INK); c.setFont('YaHeiB',19); c.drawString(42,h-78,title)
    y=para(subtitle,h-93,style=SMALL)
    footer(n,w)
    return y-9

def table(data,y):
    cells=[[p('编号',HEAD),p('实验 / FID · PFID',HEAD),p('相对变化与实现',HEAD)]]
    for r in data:
        cells.append([p(r['id']),p(f"{r['label']}\nFID {r['FID']:.4f} / PFID {r['Patch_FID']:.4f}"),p(r['difference'])])
        # Explicit line break avoids relying on whitespace interpretation.
        cells[-1][1]=Paragraph(escape(r['label']).replace('x₀','x0')+f"<br/><font size='8' color='#556577'>{r['FID']:.4f} / {r['Patch_FID']:.4f}</font>",STYLE)
    t=Table(cells,colWidths=[36,164,A4[0]-284],hAlign='LEFT')
    t.setStyle(TableStyle([
        ('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f1f8')),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fb')]),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#b9cbdc')),
    ]))
    _,h=t.wrap(A4[0]-84,1000)
    if y-h<48: raise RuntimeError(f'Table overflow: remaining {y-h}')
    t.drawOn(c,42,y-h)
    return y-h-16

# Page 1: high-resolution overview, with room for all labels.
w,h=landscape(A3)
iw=w-48; ih=iw*2160/3600
c.drawImage(str(SRC/'FID_PFID_全部实验.png'),24,(h-ih)/2+10,width=iw,height=ih)
footer(1,w)
c.showPage()

y=start_page(2,'实验配置 01 / 基线与先验','编号对应总览图；每行数值依次为 FID / PFID，越低越好。G160是160次学生更新，不是160轮数据集。')
y=para('共同口径：20组自训模型均取G160，每组评测10,000张。编号1-4使用999/919/499/199；其余自训组使用999/749/499/249。公开SDXL、DMD2只作为质量参照，训练预算不同。',y)
y=table(rows[:10],y)
c.showPage()

y=start_page(3,'实验配置 02 / 监督与GAN消融','以下改动应按指定基线比较；一次改变多个训练环节的组，不能只归因于某一个机制。')
y=table(rows[10:19],y)
y=para('“1%学生→fake”指从G1开始，每次学生更新后执行共享UNet参数 fake = 0.99 × fake + 0.01 × student；不混合fake专用阶段嵌入及Adam状态。它是对fake参数的注入，不是推理学生的EMA。',y,style=SMALL)
c.showPage()

y=start_page(4,'实验配置 03 / 新增结果与解读','本页补充已核验完成的三组，并区分观察结果和可以成立的消融结论。')
y=table(rows[19:],y)
for title,body in [
 ('当前自训最低FID：12号，22.1288', 'C＋1%只保留首步PCM和末步学生DM，PFID为28.0241。注意：该组fake仍有中段更新，不能解释为完全移除了中段训练。'),
 ('当前自训最低PFID：15号，25.6695', '额外去噪GAN组的FID为25.6672。局部图像分布指标改善并没有同步产生最低的全图FID。'),
 ('局部加噪旧组：19、20号表现较好', '归一化与原始转角组FID分别为22.9175、23.0569，PFID分别为26.6626、25.8994。它们与A/B/C还同时存在加噪范围、fake阶段轮转和先验对齐方式的差别，不能仅据此证明转向先验更好。'),
 ('A的PCM×10加入1%混合：变化很小', '18号对比13号：FID降低约0.0122，PFID升高约0.0902；单次结果不足以支持有明确提升。'),
]:
    y=para(title,y,style=HEAD)
    y=para(body,y)
y=para('指标定义与限制：PFID沿用项目的中心299×299裁剪Patch FID，不能代表全图所有区域的纹理。FID与PFID均不直接衡量提示词语义一致性。此图未纳入尚无完整分数的PCM×10＋末端低噪声先验训练。',y,style=SMALL)
y=para('数据来源：既有完整COCO10K结果及项目汇总；18-20号于2026-11-13直接核对实验室服务器的complete.json，均为10,000项完成记录。逐项来源与原始精度保存在配套图表目录的results_and_recipes.json中。',y,style=SMALL)
if y<46: raise RuntimeError(f'Page4 overflow {y}')
c.save()
print(OUT)
