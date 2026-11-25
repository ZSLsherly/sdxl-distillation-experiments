from pathlib import Path
from html import escape
from PIL import Image, ImageDraw, ImageFont
import json, io, statistics
root=Path('output/sdxl-crossdistill/a800_BC_paired_quota_20261102')
out=root/'final_comparison';out.mkdir(exist_ok=True)
arms=['B','C','uniform_control']
read=lambda p: json.loads(p.read_text(encoding='utf8'))
def samples(arm):
    p=next((root/'completed'/arm/'eval').glob('*_g000160_*/rank0.json'))
    return p.parent,read(p)['samples']
runs=[samples(a) for a in arms]
font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
large=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
cards=[]
for i in range(8):
    board=Image.new('RGB',(920,457),'white'); d=ImageDraw.Draw(board)
    ref=runs[0][1][i]
    for j,(path,rows) in enumerate(runs):
        rec=rows[i];assert all(rec[k]==ref[k] for k in ('split','original_index','prompt','seed','image'))
        fname=rec['image']; folder=root/'completed'/'preview'; stem=Path(fname).stem
        if j<2:
            with Image.open(path/fname) as im:
                assert im.size==(1024,1024)
                views=[im.resize((360,360),Image.Resampling.LANCZOS), im.crop((362,362,661,661))]
            # Match the preview encoding across arms before composing the board.
            for n,v in enumerate(views):
                b=io.BytesIO();v.convert('RGB').save(b,'JPEG',quality=92);b.seek(0); views[n]=Image.open(b).copy()
        else:
            views=[Image.open(folder/f'{stem}_{suffix}.jpg') for suffix in ('full','crop')]
        left=j*305+4
        d.text((left,4),('B · 高噪声','C · 高低噪声','无先验')[j],fill='#18354b',font=large)
        board.paste(views[0].resize((217,217),Image.Resampling.LANCZOS),(left+39,40))
        board.paste(views[1].resize((185,185),Image.Resampling.LANCZOS),(left+55,266))
    name=f'{stem}.jpg';board.save(out/name,quality=94)
    cards.append((name,ref))
for split in ['train','holdout']:
    picked=[item for item in cards if item[1]['split']==split]; assert len(picked)==4
    sheet=Image.new('RGB',(920,4*345),'white')
    for i,(name,rec) in enumerate(picked):
        im=Image.open(out/name).resize((650,323),Image.Resampling.LANCZOS)
        sheet.paste(im,(185,i*345));ImageDraw.Draw(sheet).text((3,i*345+30),f"{split}\nID {rec['original_index']}",fill='black',font=font)
    sheet.save(out/f'{split}_overview.jpg',quality=92)
diagnostics=read(root/'loss_diagnostics'/'summary.json')
metrics=[]
for name,arm in zip(('B','C','无先验'),arms):
    s=diagnostics[name]; v=read(root/'completed'/f'checkpoint_validation_{arm}.json')
    assert v['strict_model_load']=='passed' and v['all_tensors_finite'] and s['G']==160 and s['fake_updates']==320
    metrics.append('<tr>'+''.join(f'<td>{x}</td>' for x in (name,*[f"{s['windows'][w]['4']['loss']:.4f}" for w in ('1-40','41-80','81-120','121-160')],s['fake_clipped']))+'</tr>')
findings=(out/'visual_notes.txt').read_text(encoding='utf8') if (out/'visual_notes.txt').exists() else ''
page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>三组 G160 对照</title><style>body{font:16px/1.65 system-ui;max-width:980px;margin:32px auto;padding:0 16px;color:#213746}img{max-width:100%}figure{margin:2rem 0}th,td{border-bottom:1px solid #ddd;text-align:right;padding:.4rem .8rem}h2{margin-top:2rem}small{color:#555}</style>
<h1>高噪声先验、双区间先验与无先验 · G160</h1><p>三组均独立从原始 SDXL 出发。训练数据为同一 10K 子集，单卡有效 batch 64，每组学生更新 160 次，fake 更新 320 次。三组全部周期的中段及末步配对、日志数值与最终 8 张样图已核对。三个最终学生权重均通过 CPU 严格加载与有限性检查。</p>
<p>B 在每段合法范围内给高噪声 900–979 分配 30% 概率；C 在末步把这 30% 分给 20–100 与 900–979 的并集，中段与 B 一致。无先验在各段合法区间均匀采样。三组均关闭 GAN。</p>
<h2>末步 fake 的训练损失</h2><table><tr><th>组</th><th>G1–40</th><th>G41–80</th><th>G81–120</th><th>G121–160</th><th>fake 梯度范数 &gt; 1 次数（共 320）</th></tr>'''+''.join(metrics)+'''</table><p>各值是末步训练回归误差的区间均值。时间步分布和学生生成分布随实验而异，不能把跨组的高低直接解释为跟踪能力。第 2、3 步拟合带噪边界，末步拟合干净终点。学生 DM 替代损失、裁剪前梯度和分阶段 fake 图在<a href="../loss_diagnostics/report.html">训练曲线与分阶段诊断</a>。</p>
<h2>固定样本</h2><p>每行同一提示词、同一初始种子 102。列从左至右为 B、C、无先验。上排整图，下排同一中心 299×299 区域。保留全部 4 个训练提示和 4 个留出提示。</p><p>'''+escape(findings)+'''</p>'''
for name,rec in cards:
    page+=f"<figure><figcaption><b>{'留出' if rec['split']=='holdout' else '训练'} · {rec['original_index']}</b>　{escape(rec['prompt'])}</figcaption><img loading='lazy' src='{name}'></figure>"
(out/'report.html').write_text(page+'</html>',encoding='utf8')
