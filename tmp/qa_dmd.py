from pathlib import Path
import json, collections
from PIL import Image
from pypdf import PdfReader
from reportlab.pdfbase.ttfonts import TTFont
root=Path(__file__).resolve().parent.parent
folder=root/'tmp/pdfs/dmd_render'
fs=sorted(folder.glob('page-*.png'))
for i in range(0,len(fs),2):
 ims=[Image.open(p).convert('RGB') for p in fs[i:i+2]]
 canvas=Image.new('RGB',(ims[0].width*2+16,ims[0].height),'#dfe4e8')
 canvas.paste(ims[0],(0,0))
 if len(ims)>1:canvas.paste(ims[1],(ims[0].width+16,0))
 canvas.save(folder/f'pair-{i//2+1:02}.png')
d=json.loads((root/'tmp/dmd_report_content.json').read_text(encoding='utf-8'))
font=TTFont('Y','C:/Windows/Fonts/msyh.ttc',subfontIndex=0)
chars=collections.Counter(json.dumps(d,ensure_ascii=False))
print('Missing glyph codepoints:',[(hex(ord(c)),n) for c,n in chars.items() if ord(c)>127 and ord(c) not in font.face.charToGlyph])
r=PdfReader(next((root/'output/pdf').glob('DMD*.pdf')))
for i,pg in enumerate(r.pages):
 txt=pg.extract_text()
 for j,c in enumerate(txt):
  if c=='\x00':print('Null',i+1,ascii(txt[max(0,j-25):j+25]))
print('Pages',len(r.pages),'Bytes',next((root/'output/pdf').glob('DMD*.pdf')).stat().st_size)
print('Links',sum(len(p.get('/Annots',[])) for p in r.pages))
print('Chinese characters',sum('\u4e00'<=c<='\u9fff' for c in json.dumps(d,ensure_ascii=False)))
