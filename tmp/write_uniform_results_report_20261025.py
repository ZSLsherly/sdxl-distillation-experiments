from pathlib import Path
import json,html
root=Path(__file__).resolve().parents[1];out=root/'output/sdxl-crossdistill/uniform10k_results_20261025'
r=json.loads((out/'results.json').read_text());s=r['status'];w=r['window_metrics']
def table(headers,rows):
 return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
body=f'''<h1>均匀四步 · 末步 GAN+DMD · 10K 双卡结果</h1><p>2026-10-25 实时核验 · 完成于 2026-10-24 06:42（北京时间）</p>
<p class="ok"><strong>训练正常完成 G160，四个权重节点与最终完整续训状态齐全。固定样图相较初始四步明显更清晰，但局部伪影仍存在；目前尚不能单独证明 GAN 带来额外收益。</strong></p>
<h2>运行结果</h2>'''
body+=table(['项目','结果'],[
 ('方案','999/749/499/249；PCM + DMD；仅第4步加GAN；共享fake轮转1/2/3/4'),
 ('数据与GPU','现有10K训练集；172.18.33.60，A100 40GB物理卡0、2'),
 ('批量','全局64；G1–2每卡4、累积8；G3–160每卡16、累积2'),
 ('更新完成','学生160；fake800；判别器800；GAN学生更新53；R1共50次'),
 ('学生样本曝光','160×64=10,240；其中GAN学生更新53×64=3,392次样本曝光'),
 ('续训段墙钟时间',f"{r['guard']['wall_seconds']/3600:.2f} 小时，含G2恢复、评估与最终存档；不含前段及batch调优"),
 ('稳定段完整cycle均值',f"{r['steady_mean_cycle_seconds']:.2f} 秒，统计G10–160，含5次fake/D＋1次G"),
 ('显存峰值',f"allocated {s['peak_allocated_GiB']:.2f} GiB / reserved {s['peak_reserved_GiB']:.2f} GiB（单卡两rank取最大）"),
 ('存档','G40、G80、G120、G160；最终完整状态complete标记、模型/优化器/RNG文件齐全'),
 ('运行健康','两rank日志计数连续、损失/梯度均有限、退出码0；监测未发现他人进程进入指定卡'),
])
body+='''<p>本次训练进程已退出。当前服务器GPU上的占用不代表本任务仍在训练。完整状态核验依据保存日志、complete标记与文件齐全性，没有重新加载约58GiB状态进行额外GPU测试。</p>
<h2>图像效果：先看留出样本</h2>
<p>同一提示词、同一种子102。G0是<strong>原始SDXL权重＋本实验四步VP/DDIM采样＋CFG1</strong>，不是原始SDXL的50步质量上限。下图列出固定的全部4条留出提示词，未按结果筛选。另有全部4条训练提示词对照。</p>
<a href="holdout_progress.jpg"><img src="holdout_progress.jpg" alt="全部留出样本的五节点对照"></a>
<p><strong>观察：</strong>G0→G80的清晰化明显，房屋、窗框、室内家具及婚礼场景的主体与边缘更明确；G120→G160主要是局部细节、颜色和构图变化，并非所有例子都单调提升。部分人物五官、器物边缘和字样仍不自然。</p>
<h2>纹理与局部伪影</h2>
<p>下面直接取原始1024px图的中心256×256像素，没有放大插值。仍能看到细密条纹、块状色阶及部分脸部/器物细节瑕疵。清晰度改善不等于自然纹理已经解决。各节点构图有变化，同位置裁剪不是语义配准对照。</p>
<a href="holdout_center_crops.png"><img src="holdout_center_crops.png" alt="原生像素局部对照"></a>
<details><summary>全部训练样本对照与局部裁剪</summary><a href="train_progress.jpg"><img src="train_progress.jpg"></a><a href="train_center_crops.png"><img src="train_center_crops.png"></a></details>
<h2>日志信号：数值正常，但不是质量评分</h2>
<a href="training_curves.png"><img src="training_curves.png" alt="训练诊断曲线"></a>'''
body+=table(['诊断量','G1–40均值','G61–100均值','G121–160均值'],[
 ('第2步 DMD surrogate',*[f"{w[k]['DMD_by_stage']['2']:.4f}" for k in ('early','middle','late')]),
 ('第3步 DMD surrogate',*[f"{w[k]['DMD_by_stage']['3']:.4f}" for k in ('early','middle','late')]),
 ('第4步 DMD surrogate',*[f"{w[k]['DMD_by_stage']['4']:.4f}" for k in ('early','middle','late')]),
 ('PCM loss（未乘系数10）',*[f"{w[k]['PCM_loss']:.6f}" for k in ('early','middle','late')]),
 ('GAN G loss（仅启用GAN的更新）',*[f"{w[k]['GAN_loss']:.4f}" for k in ('early','middle','late')]),
 ('D hinge',*[f"{w[k]['D_hinge']:.4f}" for k in ('early','middle','late')]),
])
body+='''<p>DMD surrogate按阶段分开后整体下降，但这是归一化的训练代理损失，不能当FID。PCM从早期约0.00030升至后期约0.00071，约2.4倍，是后续值得关注的目标权衡信号；绝对值仍小，不能据此断言训练崩溃。梯度范数周期起伏与第2/3/4阶段轮转有关。</p>
<h2>现在能下什么结论</h2><ul>
<li>可以确认：新版算法与双卡micro16运行链路跑完，留出样图也明显好于同采样器的原始四步输出，未看到数值崩溃证据。</li>
<li>不能确认：GAN比纯DMD更优、G160优于所有中间节点、没有过拟合或多样性损失。只有4条固定留出提示词和同一个种子，证据太少。</li>
<li>本次训练的8张周期样图不是COCO10K benchmark。这组权重尚未在本任务中完成完整COCO10K FID/Patch-FID，不能借用旧组分数。</li>
<li>训练采用10K数据不意味着GAN学生单独遍历了10K；它实际只更新53次。训练量仍属于试验规模。</li>
<li>判断GAN额外收益，需要相同采样点、修正后的fake策略、数据和预算下的DMD-only对照。旧实验同时改变过采样/训练实现，不能直接做因果归因。</li></ul>
<p>下一步最有价值的是对现有节点做同协议质量评测，并补匹配的DMD-only对照；本轮仅读取和整理结果，没有启动新训练或GPU评测。</p>
<h2>原图与证据</h2>'''
for split in ('holdout','train'):
 body+=f'<h3>{split}</h3><ul>'
 cases=sorted([x for x in r['evals']['0']['samples'] if x['split']==split],key=lambda x:x['original_index'])
 for case in cases:
  links=' · '.join(f'<a href="images/g{g:03d}/{html.escape(case["image"])}">G{g}</a>' for g in (0,40,80,120,160))
  body+=f'<li>#{case["original_index"]} {html.escape(case["prompt"])}<br>{links}</li>'
 body+='</ul>'
body+='<p><a href="results.json">完整日志与存档核验记录</a> · <a href="training_curves.pdf">曲线PDF</a> · <a href="initial_vs_final.jpg">简明前后对照</a></p>'
doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>10K双卡G160实验结果</title><style>body{max-width:1320px;margin:40px auto;padding:0 24px;font:17px/1.75 system-ui,sans-serif;color:#24354b;background:#f8fafc}h1{font-size:30px}h2{font-size:24px;color:#175575;margin-top:32px}table{border-collapse:collapse;width:100%;background:#fff}th,td{padding:11px;border:1px solid #d6deea;text-align:left}a{color:#1769a4}img{width:100%;height:auto;margin:15px 0;border:1px solid #d6deea}.ok{padding:18px;background:#e5f2eb;border-left:5px solid #3a815b}details{background:#edf1f5;padding:16px}summary{cursor:pointer}li{margin:8px 0}td,p,li{overflow-wrap:anywhere}</style>'+body+'</html>'
(out/'report.html').write_text(doc,encoding='utf-8')
print(out/'report.html')
