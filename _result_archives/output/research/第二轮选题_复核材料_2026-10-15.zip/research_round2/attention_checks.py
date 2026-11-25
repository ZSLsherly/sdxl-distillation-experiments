"""Exact attention counterexamples. CPU only; no video-model validation."""
from pathlib import Path
import json, math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT=Path(__file__).resolve().parent

def attention(q,k,v):
    logits=np.asarray(k)@np.asarray(q)/math.sqrt(len(q))
    m=float(logits.max())
    z=np.exp(logits-m)
    return float(z@np.asarray(v)/z.sum()),float(m+np.log(z.sum()))

q=np.array([1.0]); k=np.zeros((2,1)); v=np.array([0.,2.])
retk=np.array([[math.log(2.)]]); retv=np.array([0.])
base_o,base_lse=attention(q,k,v)
full_o,_=attention(q,np.vstack([k,retk]),np.r_[v,retv])
rows=[]
for factor in [0.1,1.,9.,100.]:
    predk=k+math.log(factor)
    part,lse=attention(q,predk,v)
    full,_=attention(q,np.vstack([predk,retk]),np.r_[v,retv])
    rows.append({'relative_partition_factor':factor,'cache_only_output':part,
        'output_only_loss':(part-base_o)**2,'lse_error':lse-base_lse,
        'full_attention_output':full,'full_attention_abs_error':abs(full-full_o)})

# A loss measured on one query direction does not cover an unseen orthogonal one.
qtrain=np.array([math.sqrt(2),0.]); qtest=np.array([0.,math.sqrt(2)])
kt=np.zeros((2,2)); kp=np.array([[0.,math.log(9.)],[0.,math.log(9.)]])
kr=np.array([[math.log(2.),math.log(2.)]])
shift={}
for name,qq in [('train_query',qtrain),('unseen_query',qtest)]:
    ot,lt=attention(qq,kt,v); op,lp=attention(qq,kp,v)
    ft,_=attention(qq,np.vstack([kt,kr]),np.r_[v,retv])
    fp,_=attention(qq,np.vstack([kp,kr]),np.r_[v,retv])
    shift[name]={'partial_output_error':op-ot,'lse_error':lp-lt,
                 'teacher_full_output':ft,'predicted_full_output':fp,'abs_error':abs(fp-ft)}
results={'status':'Exact CPU attention examples; not a new theorem and not video evidence.',
         'source_of_output_lse_contract':'https://arxiv.org/abs/2603.27819',
         'base_partial_output':base_o,'base_full_output':full_o,
         'mass_shift_rows':rows,'query_distribution_shift':shift}
(OUT/'attention_checks.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')

im=Image.new('RGB',(1600,780),'white'); d=ImageDraw.Draw(im)
fontpath='C:/Windows/Fonts/msyh.ttc'; boldpath='C:/Windows/Fonts/msyhbd.ttc'
def font(n,b=False):return ImageFont.truetype(boldpath if b else fontpath,n)
def text(x,y,s,n=25,color='#172B42',b=False):d.text((x,y),s,font=font(n,b),fill=color)
text(45,22,'缓存行为检查：匹配局部输出仍然可能写错状态',35,b=True)
text(45,76,'解析 CPU 反例；output + LSE 契约来自 KVSculpt，非本报告的新定理或视频结果。',23,'#586B7D')
text(45,140,'1  注意力总质量不能漏掉',29,b=True)
text(45,184,'局部输出都是 1；加入另一个历史块后，完整输出不同。',23)
left,top,bottom=95,260,610
for j,y in enumerate([0,.25,.5,.75,1.]):
    yy=bottom-y*(bottom-top)
    d.line((left,yy,740,yy),fill='#D6E1E8',width=2)
    text(48,yy-15,f'{y:g}',21,'#586B7D')
for i,r in enumerate(rows):
    xx=160+i*145; yy=bottom-r['full_attention_output']*(bottom-top)
    d.rectangle((xx,yy,xx+80,bottom),fill='#0C8C90' if i==1 else '#D88A4E')
    text(xx+3,yy-34,f"{r['full_attention_output']:.3f}",24,b=True)
    text(xx-8,bottom+15,str(r['relative_partition_factor']),23)
basey=bottom-.5*(bottom-top)
d.line((left,basey,740,basey),fill='#172B42',width=3)
text(110,678,'横轴：缓存 exp-logit 总质量的倍数；基准输出 = 0.5',22,'#586B7D')
text(840,140,'2  还要检查未来查询的分布变化',29,b=True)
texts=[('训练查询方向','局部输出误差 = 0','LSE 误差 = 0','完整输出误差 = 0'),
       ('未见的正交查询','局部输出误差 = 0','LSE 误差 = log(9)','完整输出误差 = 0.4')]
for i,lines in enumerate(texts):
    y=225+i*185
    d.rounded_rectangle((840,y,1550,y+155),radius=12,fill='#EDF7F6' if i==0 else '#FFF4E9')
    text(860,y+15,lines[0],26,b=True)
    for j,s in enumerate(lines[1:]):text(860,y+53+j*29,s,23)
text(840,640,'含义：训练局部 loss 小，不保证闭环长视频稳定。',23,'#586B7D')
text(840,675,'需要实际 rollout 查询、分支查询及完整 reader 检查。',23,'#586B7D')
im.save(OUT/'attention_checks.png')
print(json.dumps({'rows':rows,'query_shift':shift},ensure_ascii=False))
