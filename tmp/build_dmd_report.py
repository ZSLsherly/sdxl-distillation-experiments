from pathlib import Path
from xml.sax.saxutils import escape
import json, re
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Flowable
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'output'/'pdf'
OUT.mkdir(parents=True,exist_ok=True)
pdfmetrics.registerFont(TTFont('YaHei','C:/Windows/Fonts/msyh.ttc',subfontIndex=0))
pdfmetrics.registerFont(TTFont('YaHeiBold','C:/Windows/Fonts/msyhbd.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('YaHei',normal='YaHei',bold='YaHeiBold',italic='YaHei',boldItalic='YaHeiBold')
INK=colors.HexColor('#162539'); BLUE=colors.HexColor('#245C7B'); GREY=colors.HexColor('#596778')
W,H=A4
styles={
 'body':ParagraphStyle('body',fontName='YaHei',fontSize=10,leading=16.2,textColor=INK,spaceAfter=8,wordWrap='CJK'),
 'small':ParagraphStyle('small',fontName='YaHei',fontSize=8.5,leading=13.2,textColor=GREY,spaceAfter=6,wordWrap='CJK'),
 'h1':ParagraphStyle('h1',fontName='YaHeiBold',fontSize=19,leading=26,textColor=INK,spaceAfter=16,wordWrap='CJK'),
 'h2':ParagraphStyle('h2',fontName='YaHeiBold',fontSize=12.2,leading=18,textColor=INK,spaceBefore=8,spaceAfter=7,wordWrap='CJK',keepWithNext=True),
 'title':ParagraphStyle('title',fontName='YaHeiBold',fontSize=30,leading=42,textColor=INK,spaceAfter=19,wordWrap='CJK'),
 'deck':ParagraphStyle('deck',fontName='YaHei',fontSize=13,leading=21,textColor=GREY,spaceAfter=22,wordWrap='CJK'),
 'cell':ParagraphStyle('cell',fontName='YaHei',fontSize=8.4,leading=12.5,textColor=INK,wordWrap='CJK'),
 'thead':ParagraphStyle('thead',fontName='YaHeiBold',fontSize=8.5,leading=12.5,textColor=colors.white,wordWrap='CJK'),
 'ref':ParagraphStyle('ref',fontName='YaHei',fontSize=8.8,leading=13.8,textColor=INK,spaceAfter=11,wordWrap='CJK'),
}
def rich(s):
 s=escape(s).replace('\n','<br/>')
 s=s.replace('ₜ','<sub>t</sub>').replace('ᵣ','<sub>r</sub>').replace('₀','<sub>0</sub>').replace('∇','grad')
 s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
 s=re.sub(r'\[(\d{2})\]',lambda m:f'<link href="#ref{m[1]}" color="#245C7B">[{m[1]}]</link>',s)
 return s
def para(s,style='body'): return Paragraph(rich(s),styles[style])

class Scheme(Flowable):
 def __init__(self): Flowable.__init__(self); self.width=491; self.height=165
 def draw(self):
  c=self.canv
  def box(x,y,w,h,title,sub):
   c.setFillColor(colors.HexColor('#EEF3F6')); c.setStrokeColor(colors.HexColor('#C6D5DF')); c.roundRect(x,y,w,h,5,fill=1,stroke=1)
   c.setFillColor(INK); c.setFont('YaHeiBold',10); c.drawCentredString(x+w/2,y+h-19,title)
   c.setFont('YaHei',8); c.drawCentredString(x+w/2,y+13,sub)
  def arrow(x1,y1,x2,y2):
   c.setStrokeColor(BLUE); c.setLineWidth(1); c.line(x1,y1,x2,y2)
   if x2>x1: c.line(x2,y2,x2-5,y2+3); c.line(x2,y2,x2-5,y2-3)
  box(0,95,112,53,'学生生成器 G','噪声和文本 → 少步输出')
  box(147,95,112,53,'重新加噪 x_t','随机噪声尺度 t')
  box(295,107,188,45,'冻结教师 score','目标分布的局部方向')
  box(295,45,188,45,'可训练 fake score','跟踪当前学生的分布')
  arrow(112,121,147,121); arrow(259,127,295,127)
  c.line(277,121,277,68); arrow(277,68,295,68)
  c.setFillColor(INK); c.setFont('YaHei',9); c.drawString(0,43,'学生更新使用两个 score 的差；fake score 用学生样本做去噪训练。')
  c.setFont('YaHeiBold',9); c.drawString(0,17,'推理阶段仅保留学生生成器，以及文本编码器和解码器。')

def table(headers,rows,widths=None):
 widths=widths or [491/len(headers)]*len(headers)
 data=[[para(x,'thead') for x in headers]]+[[para(str(x),'cell') for x in row] for row in rows]
 t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),BLUE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F2F5F7')]),('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#D7DFE5')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
 return t

def chrome(c,doc):
 c.saveState(); c.setFillColor(GREY); c.setFont('YaHei',8)
 if doc.page>1: c.drawString(52,H-31,'DMD 图像与视频生成推理加速综述')
 c.drawString(52,27,'调研截止 2026 年 10 月 13 日')
 c.drawRightString(W-52,27,str(doc.page))
 c.restoreState()

data=json.loads((ROOT/'tmp'/'dmd_report_content.json').read_text(encoding='utf-8'))
story=[]
for page in data['pages']:
 if story: story.append(PageBreak())
 if page.get('cover'):
  story.extend([Spacer(1,24),para('DMD 领域进展综述','title'),para('图像与视频生成的少步推理加速\n技术脉络 关键证据 研究与落地建议','deck')])
 else: story.append(para(page['title'],'h1'))
 for block in page['blocks']:
  typ=block['type']
  if typ in ['p','small','h2']: story.append(para(block['text'],{'p':'body','small':'small','h2':'h2'}[typ]))
  elif typ=='table':
   story.append(table(block['headers'],block['rows'],block.get('widths')));story.append(Spacer(1,10))
  elif typ=='scheme': story.append(Scheme());story.append(Spacer(1,8))
  elif typ=='refs':
   for key in block['keys']:
    r=data['refs'][key]
    s=f'<a name="ref{key}"/>[{key}] <b>{escape(r["title"])}</b><br/>{escape(r["meta"])}<br/><link href="{escape(r["url"])}" color="#245C7B">{escape(r["url"])}</link>'
    if r.get('code'):s+=f'<br/>代码或项目 <link href="{escape(r["code"])}" color="#245C7B">{escape(r["code"])}</link>'
    story.append(Paragraph(s,styles['ref']))
pdf=OUT/'DMD图像与视频推理加速综述_2026-10-13.pdf'
doc=SimpleDocTemplate(str(pdf),pagesize=A4,rightMargin=52,leftMargin=52,topMargin=57,bottomMargin=49,title='DMD图像与视频推理加速综述',author='研究综述',pageCompression=1)
doc.build(story,onFirstPage=chrome,onLaterPages=chrome)
print(pdf)
