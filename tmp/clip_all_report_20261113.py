import json,html,fcntl
from pathlib import Path
import numpy as np
ROOT=Path('/data3/shenglin.zeng/clip_missing_20261113')
NAMES={'dmd_only':'旧四点·纯DMD','gan4':'旧四点·末步纯GAN','joint4':'旧四点·末步GAN+DMD','joint34':'旧四点·3/4步GAN+DMD','joint4_uniform':'均匀四点·旧clean目标·末步GAN+DMD','A':'A·无先验','B':'B·高噪声先验','C':'C·高低噪声先验','C_mix1':'C＋1%','DM234_fake3':'C＋1%·DM234/fake3','DM1_mid4_fake3':'C＋1%·首步DM＋随机中段DM＋末步DM','DM1_uniform':'无先验·首步DM＋随机中段DM＋末步DM＋1%','PCM_DM4_G160':'PCM＋仅末步学生DM（保留中段fake）','A_pcm10':'A·PCM×10','A_pcm10_patchgan':'A·PCM×10＋末步PatchGAN','A_mid_clean_x0':'A·中段clean x0','A_pcm10_mix1':'A·PCM×10＋1%','H20_extra_GAN':'H20·额外去噪GAN','normalized_u_g160':'旧局部区间·归一化转向','raw_angle_g160':'旧局部区间·原始转角','sdxl':'原始SDXL·50步','dmd2':'官方DMD2·4步'}
def main():
 with (ROOT/'report.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  data=json.loads((ROOT/'existing_results.json').read_text());pending=[]
  for a in NAMES:
   if a in data['metrics']:continue
   p=ROOT/'arms'/a/'results.json'
   if not p.exists():pending.append(a);continue
   d=json.loads(p.read_text());assert d['cases']==data['cases'] and d['revision']==data['revision'] and d['n']==1024
   data['metrics'].update(d['metrics']);data['per_case'].update(d['per_case'])
  data['pending']=pending;data['labels']=NAMES
  rows=[];lines=['COCO1024 CLIP补测：同提示词、同种子、OpenAI ViT-B/32，cosine×100。',f"完成 {len(data['metrics'])}/22（含两个公开参照）。"]
  baseline=np.array(data['per_case']['A']['alignment'])
  for a in NAMES:
   m=data['metrics'].get(a)
   if m:
    v=m['cosine_x100'];ci=''
    if a in data['per_case']:
     diff=np.asarray(data['per_case'][a]['alignment'])-baseline;rng=np.random.default_rng(20261103)
     means=np.concatenate([diff[rng.integers(0,1024,(200,1024))].mean(1) for _ in range(10)])
     bounds=np.percentile(means,[2.5,97.5]);ci=f'{diff.mean():+.4f} [{bounds[0]:+.4f}, {bounds[1]:+.4f}]'
    cells=[NAMES[a],f'{v:.6f}',f"{m['top1_correct']}/1024",f"{m['top5_correct']}/1024",ci,'完成']
   else:
    status=ROOT/'arms'/a/'status.json';progress='待输入/排队'
    if status.exists():progress=f"{json.loads(status.read_text())['done']}/1024"
    cells=[NAMES[a],'—','—','—','—',progress]
   rows.append('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in cells)+'</tr>');lines.append(' | '.join(cells))
  (ROOT/'results_all.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
  (ROOT/'评测记录.txt').write_text('\n'.join(lines),encoding='utf-8')
  (ROOT/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px Arial;max-width:1250px;margin:40px auto}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>全部已完成G160模型：COCO1024 CLIP补测</h1><p>'+html.escape(lines[1])+'</p><p>沿用20261103固定1024样本。CLIP越高越好；Top1/Top5为同批文本检索，重复caption视为等价。相对A差值附配对bootstrap 95%区间，仅反映此固定样本集合的不确定性，不代替多种子训练。</p><p>只使用172.18.32.81物理5号卡；所有物理4号卡都不使用。复用已有PNG，仅保存新增图像的CLIP特征。不是新增FID评测。</p><table><tr><th>实验</th><th>CLIP↑</th><th>Top1</th><th>Top5</th><th>相对A [95%CI]</th><th>进度</th></tr>'+''.join(rows)+'</table>',encoding='utf-8')
if __name__=='__main__':main()
