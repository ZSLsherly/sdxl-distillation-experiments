import json,html
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

root=Path('output/sdxl-crossdistill/original_sdxl_control')
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',21)
small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',16)
methods=[('base_50_cfg8','原始 SDXL · 50步 CFG8'),('base_4_cfg1','原始 SDXL · 同四步 CFG1'),('before_v2','本轮前 · 已训练12次'),('after_A','本轮后 A · TM+DMD'),('after_C','本轮后 C · 末步GAN+DMD')]
canvas=Image.new('RGB',(1800,840),'white');draw=ImageDraw.Draw(canvas)
sections=[]
for row,idx in enumerate([113,118]):
    meta=json.loads((root/f'{idx}_metadata.json').read_text())
    for col,(method,title) in enumerate(methods):
        x=col*360;y=row*420
        im=Image.open(root/f'{idx}_{method}.png').convert('RGB');im.thumbnail((352,352))
        canvas.paste(im,(x+4,y+45));draw.text((x+8,y+10),title,fill='black',font=small)
        draw.text((x+8,y+398),f'样本 {idx} · seed 102',fill='black',font=small)
    gallery=[]
    allmethods=methods+[('base_4_cfg8','原始 SDXL · 同四步 CFG8'),('after_B','本轮后 B · 末步仅GAN'),('after_D','本轮后 D · 第3/4步GAN')]
    for method,title in allmethods:
        src=f'{idx}_{method}.png'
        im=Image.open(root/src).convert('RGB');im.crop((384,384,640,640)).save(root/f'{idx}_{method}_crop.png')
        gallery.append(f'<article><h3>{title}</h3><a href="{src}"><img src="{src}"></a><p>中心 256×256 原尺寸裁剪</p><img class="crop" src="{idx}_{method}_crop.png"></article>')
    sections.append(f'<section><h2>样本 {idx} / seed 102</h2><p>{html.escape(meta["prompt"])}</p><div class="gallery">'+''.join(gallery)+'</div></section>')
canvas.save(root/'comparison.png')
(root/'report.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>原始 SDXL 与训练前后对照</title><style>body{font-family:system-ui,"Microsoft YaHei";margin:32px;background:#f4f5f7;color:#20252d}h1{font-size:28px}p{line-height:1.7}section{margin:30px 0}.gallery{display:grid;grid-template-columns:repeat(4,minmax(240px,1fr));gap:16px}article{background:white;padding:12px;border-radius:10px}img{width:100%;height:auto}img.crop{width:256px;max-width:100%;image-rendering:auto}h3{font-size:16px}a{color:#1765a6}</style><h1>原始 SDXL 与训练前后对照</h1><p>仅推理，没有新训练。相同 LAION 留出提示词、随机种子102、1024分辨率、同一FP32 VAE整图解码。点击图片可看原始1024图。</p><p><b>公平控制：</b>原始SDXL四步CFG1与学生使用相同VP/DDIM时间步999→919→499→199→clean、相同float32初始噪声及条件UNet调用。原始50步CFG8采用模型原生pipeline采样器，是正常生成质量参考，不能单独用于归因训练退化。另给四步CFG8控制。</p><p><b>初始化：</b>“本轮前”已经进行12次此前的学生更新；本轮后为各自独立模型再更新24次。A仍包含高噪声TM。所有阶段共享UNet参数。“原始”指本地官方SDXL Base权重。</p>'''+''.join(sections)+'</html>',encoding='utf-8')
print(root.resolve()/'report.html')
