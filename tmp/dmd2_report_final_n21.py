from pathlib import Path
import json,html,hashlib,statistics
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
r=Path('output/sdxl-crossdistill/dmd2_matched_20261119/final_results_20261121')
cases=json.loads((r/'comparison_manifest.json').read_text());s=json.loads((r/'summary.json').read_text());probes=json.loads((r/'native_sampler_check/complete.json').read_text());assert probes['count']==16
m=[]
for x in cases:
 old=np.asarray(Image.open(r/'eval/g000160'/x['image']),dtype=np.float32)
 replay=np.asarray(Image.open(r/'native_sampler_check'/('ddim_replay_'+x['image'])),dtype=np.float32)
 m.append({'id':x['original_index'],'DDIM_replay_MAE_0_255':float(np.abs(old-replay).mean())})
(r/'sampler_check_summary.json').write_text(json.dumps({'replay_comparison':m,'native_count':8,'conclusion':'Severe colorful repetitive artifacts in deterministic DDIM are largely removed using the official stochastic sampler with the same final weights. Do not use DDIM preview images to judge this DMD2 training run.'},indent=2),encoding='utf-8')
for split in ['train','holdout','preview']:
 rows=[x for x in cases if x['split']==split] if split!='preview' else [next(x for x in cases if x['original_index']==i) for i in [5122,986]]
 fig,axs=plt.subplots(len(rows),2,figsize=(10,len(rows)*5),dpi=120)
 for i,x in enumerate(rows):
  for j,p in enumerate([x['A_image'],'native_sampler_check/native_'+x['image']]):
   axs[i,j].imshow(Image.open(r/p));axs[i,j].axis('off');axs[i,j].set_title(['A G160 / F320 / DDIM4','DMD2 G160 / F800 / official sampler'][j]+f'\n{x["split"]} #{x["original_index"]}',fontsize=10)
 fig.tight_layout(pad=.5);fig.savefig(r/f'{split}_native_comparison.jpg');plt.close(fig)
# Single sampler diagnostic, same weights and initial seed.
rows=[next(x for x in cases if x['original_index']==i) for i in [5122,986]]
fig,axs=plt.subplots(2,2,figsize=(10,10),dpi=110)
for i,x in enumerate(rows):
 for j,p in enumerate(['native_sampler_check/ddim_replay_'+x['image'],'native_sampler_check/native_'+x['image']]):
  axs[i,j].imshow(Image.open(r/p));axs[i,j].axis('off');axs[i,j].set_title(['Same G160: mismatched DDIM4','Same G160: official stochastic sampler'][j]+f'\n{x["split"]} #{x["original_index"]}')
fig.tight_layout();fig.savefig(r/'sampler_diagnostic.jpg');plt.close(fig)
css='body{font:16px/1.65 system-ui,sans-serif;background:#f5f6f8;color:#17212e;max-width:1250px;margin:32px auto;padding:0 20px}section{background:white;padding:24px;border-radius:12px;margin:20px 0}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #dde1e8;text-align:left}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}img{max-width:100%;height:auto}a{color:#1e5ca8}.note{padding:16px;background:#fff3d2;border-left:5px solid #d98c00}.case{margin:25px 0;border-top:1px solid #ddd;padding-top:20px}.small{font-size:14px;color:#526273}@media(max-width:750px){.grid{grid-template-columns:1fr}}'
body='''<h1>DMD2 10K 对照：G160 完成与采样器纠正</h1><p>核验日期：2026-11-21；服务器172.18.132.41；运行 formal_g160_restart_20261120。</p>
<section><h2>结论</h2><p><b>训练已完成，原生采样可以正常出图。此前统一DDIM四步预览中的大片彩色重复伪影，主要来自评估采样方式不匹配，不能作为“DMD2训练崩溃”的证据。</b></p><p class="note">评估适配是本次实现的问题：DMD2训练与官方推理均在中间步预测x0后重新加入独立噪声；我们原先却套用了A组的确定性DDIM跳步。现已用官方采样函数、同一份G160权重生成8张纠正后的样图，并重放8张DDIM图作为诊断。没有重新训练，也没有改动权重。</p><p>完成于北京时间 <b>11月21日02:34:37</b>；学生160次、fake800次；四rank日志有限，最终学生权重严格加载及全张量有限检查通过。训练循环约4小时05分，含周期评估、最终保存与加载检查约4小时20分；不含启动加载。单卡峰值已分配37.83GiB。</p><p><b>尚未做完整COCO10K：没有新的FID、Patch FID或CLIP benchmark分数。</b>以下为固定4训练提示词＋4留出提示词的小样本目视诊断。</p></section>
<section><h2>最终 loss 与走势</h2><table><tr><th>量</th><th>末次值</th><th>走势 / 含义</th></tr>'''
lg=s['last_student'];lf=s['last_fake']
rows=[('学生DM',lg['student_dm'],'前20G均值0.6392 → 后20G均值0.2683'),('学生GAN（原始值）',lg['student_gan'],'乘0.005后为 %.5f；原始数值大小不等于梯度占比'%(lg['student_gan']*.005)),('fake ε回归MSE',lf['fake_mse'],'前100F均值0.0271 → 后100F均值0.1442，约5.32倍'),('fake GAN（原始值）',lf['fake_gan'],'乘0.01后为 %.5f'%(lf['fake_gan']*.01)),('学生裁剪前梯度范数',lg['student_grad'],'裁剪阈值1'),('fake裁剪前梯度范数',lf['fake_grad'],'裁剪阈值1')]
for n,v,t in rows:body+=f'<tr><td>{n}</td><td>{v:.6f}</td><td>{t}</td></tr>'
body+='''</table><p class="small">学生末次来自G160/F796，fake末次来自F800。曲线只在真实学生更新时画学生loss，未把fake-only迭代中的0当作学生损失。fake曲线横轴F/5；淡线为原始值，粗线为滑动平均。</p><a href="loss_curves.png"><img src="loss_curves.png"></a><p>fake MSE在四个生成深度上均上升，值得检查跟踪能力；但这是不断变化的生成分布与随机加噪上的训练loss，不是固定留出误差，不能单凭它确认fake跟不上，更不能用它证明最终图像差。梯度总体有限，学生存在被裁剪的更新。</p></section>
<section><h2>A组与DMD2：同提示词、同初始种子</h2><p>A为无先验、PCM×1、边界DM、无GAN、无1%混合的G160/F320。DMD2为本次G160/F800，原生GAN，公共学习率等沿用A，非论文完整超参/数据规模复现。两者均四次UNet前向、1024分辨率、初始种子102；A使用DDIM4，DMD2使用官方随机再加噪采样。DMD2有额外噪声抽样，因此并非同一条轨迹。两者fake更新量和硬件不同，不能声称等计算预算。</p><p>纠正后的汽车、室内、建筑等图像主题清楚，大片彩色伪影消失；部分纹理和边缘仍需观察。固定8图不足以确定DMD2优于A或GAN提升了纹理。</p>'''
for x in cases:
 body+=f'<div class="case"><h3>{x["split"]} #{x["original_index"]} · seed102</h3><p>{html.escape(x["prompt"])}</p><div class="grid">'
 for title,p in [('原始SDXL，4步',x['paths'][0]),('A，G160',x['A_image']),('DMD2，G160，官方采样','native_sampler_check/native_'+x['image'])]:body+=f'<div><b>{title}</b><a href="{p}"><img loading="lazy" src="{p}"></a></div>'
 body+='</div></div>'
body+='''</section><section><h2>诊断：同一G160权重，更换采样器</h2><p>左侧为重新加载最终权重后复放的DDIM图，右侧为官方sample()输出。只是采样器改变，没有训练或修改模型。</p><a href="sampler_diagnostic.jpg"><img src="sampler_diagnostic.jpg"></a><details><summary>原先不匹配DDIM预览随训练变化（仅用于诊断，不能代表原生采样质量）</summary><img src="progression.jpg"></details></section>
<section><h2>证据与复核范围</h2><p>官方代码commit：8d8fa55633d47cfb81bbc7a892e7248f9518763f；原生采样直接提取 main/sdxl/test_folder_sdxl.py 的 sample() 与 get_x0_from_noise()，UNet BF16前向、VAE FP32。16张补测图生成后退出，未启动COCO或新训练。</p><p><a href="summary.json">完整数值汇总</a> · <a href="comparison_manifest.json">提示词/种子匹配清单</a> · <a href="sampler_check_summary.json">采样器复核</a> · <a href="complete.json">训练完成记录</a></p><p>原生图和训练日志使用rclone取回，逐文件SHA-256校验；学生权重仍保留在实验室服务器，本地未下载模型。</p></section>'''
(r/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DMD2 G160结果与采样器纠正</title><style>'+css+'</style>'+body+'</html>',encoding='utf-8')
print('REPORT_READY');print('DDIM_REPLAY_MAE',m)
