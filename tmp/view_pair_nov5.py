from pathlib import Path
from PIL import Image,ImageDraw
import json,html
d=Path('output/sdxl-crossdistill/semantic_pair_1024_20261105')
cases=json.loads((d/'cases.json').read_text())
arms=['A','A_pcm10','C_mix1','PCM_DM4_G160']
labels=['A (PCM x1)','A (PCM x10)','C+1% (middle DM)','C+1% (terminal DM only)']
page=['<meta charset=utf-8><style>body{max-width:1320px;margin:28px auto;font:16px system-ui}section{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}img{width:100%;display:block}figure{margin:0}p{color:#384759}</style><h1>四步 SDXL · 固定 COCO 提示词对照</h1><p>固定留图清单的前八例。每列上方为整图，下方为同坐标中心 299×299 原像素裁剪。点击整图查看 1024 原图。</p>']
overview=Image.new('RGB',(4*180,8*197),'white'); draw=ImageDraw.Draw(overview)
for j,r in enumerate(cases):
 i=r['index']; page.append(f"<h3>#{i} · {html.escape(r['prompt'])}</h3><p>seed {r['seed']}</p><section>")
 for k,(arm,label) in enumerate(zip(arms,labels)):
  f=f'{arm}/{i:05d}.png'; im=Image.open(d/f).convert('RGB'); assert im.size==(1024,1024)
  cf=f'{arm}/{i:05d}_center.png';im.crop((362,362,661,661)).save(d/cf)
  page.append(f'<figure>{label}<a href="{f}"><img src="{f}"></a><img src="{cf}"></figure>')
  overview.paste(im.resize((180,180)),(180*k,197*j+17));draw.text((180*k+1,197*j),f'{i} {label}',fill='black')
 page.append('</section>')
overview.save(d/'overview.jpg',quality=90);(d/'report.html').write_text(''.join(page),encoding='utf-8')

