from pathlib import Path
import json
import numpy as np
from PIL import Image, ImageDraw

base=Path('output/sdxl-crossdistill')
d=base/'semantic_pair_1024_20261105'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
results=read(d/'results.json')
earlier=read(base/'semantic_four_1024_20261103/results.json')
old=read(base/'semantic_old_uniform_20261104/results.json')
assert results['n']==old['n']==1024 and results['revision']==earlier['revision']==old['revision']
for other in (earlier,old):
    assert [(x['index'],x['prompt'],x['seed']) for x in other['cases']]==[(x['index'],x['prompt'],x['seed']) for x in results['cases']]
names={'A':'A，PCM×1','A_pcm10':'A，PCM×10','C_mix1':'C＋1%，中段＋末步 DM','PCM_DM4_G160':'C＋1%，仅末步学生 DM','joint4_uniform':'旧 clean＋末步 GAN+DMD'}
scores={'A':(25.6174,33.3767),'A_pcm10':(24.6412,35.2295),'C_mix1':(25.6565,32.9671),'PCM_DM4_G160':(22.1288,28.0241),'joint4_uniform':(23.4191,37.8199)}
source={k:(old if k=='joint4_uniform' else earlier if k in ('A','C_mix1') else results) for k in names}
summary=['组别 | CLIP×100 | Top1/1024 | Top5/1024 | FID | Patch FID']
for k,label in names.items():
    m=source[k]['metrics'][k]
    s=source[k]['per_case'][k]['alignment']
    assert len(s)==1024 and abs(np.mean(s)-m['cosine_x100'])<1e-4
    summary.append(f"{label} | {np.mean(s):.4f} | {m['top1_correct']} | {m['top5_correct']} | {scores[k][0]} | {scores[k][1]}")
for a,b in [('A_pcm10','A'),('PCM_DM4_G160','C_mix1'),('A_pcm10','joint4_uniform')]:
    delta=np.subtract(source[a]['per_case'][a]['alignment'],source[b]['per_case'][b]['alignment'])
    rng=np.random.default_rng(20261103)
    means=np.concatenate([delta[rng.integers(0,1024,(500,1024))].mean(1) for _ in range(40)])
    lo,hi=np.quantile(means,[.025,.975])
    summary.append(f'{names[a]} 减 {names[b]}：{delta.mean():+.4f}，配对95%区间 [{lo:+.4f}, {hi:+.4f}]')
txt='\n'.join(summary)
(d/'配对结果.txt').write_text(txt,encoding='utf-8')
from runpy import run_path
run_path('tmp/view_pair_nov5.py')
p=d/'report.html'
s=p.read_text(encoding='utf-8').replace('<h3>', '<pre style="white-space:pre-wrap">'+txt+'</pre><h3>',1)
p.write_text(s,encoding='utf-8')
panel=Image.new('RGB',(4*240,2*262),'white')
dr=ImageDraw.Draw(panel)
for j,i in enumerate((193,401)):
    for k,arm in enumerate(list(names)[:4]):
        im=Image.open(d/arm/f'{i:05d}.png')
        panel.paste(im.resize((240,240)),(240*k,j*262+22))
        dr.text((240*k+4,j*262+3),f'#{i} {arm}',fill='black')
panel.save(d/'preview.jpg',quality=93)
print(txt)
