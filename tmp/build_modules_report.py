import hashlib
import html
import json
from pathlib import Path
import re
import zipfile

root=Path(__file__).resolve().parents[1]
src=root/'sdxl_research'
out=root/'output/sdxl-crossdistill/modules_prepared_20261022'
manifest=json.loads((src/'source_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((src/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
log=(out/'evidence/cpu_tests.log').read_text()
assert 'Ran 47 tests' in log and log.rstrip().endswith('OK')
(out/'evidence/validation.json').write_text(json.dumps({
 'tests_passed':47,'visible_GPUs':0,'training_started':False,
 'remote_validation_directory':'/data5/shenglin.zeng/sdxl_crossdistill/validation/modules_prepared_20261022',
 'source_sha256':manifest,'validation_scope':'CPU tensor, gradient, routing and regression tests; not full pretrained SDXL GPU validation'
},ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(out/'sdxl_research_modules.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in list(manifest)+['source_manifest.json']:
  z.write(src/name,'sdxl_research/'+name)
 z.write(root/'sdxl_parallel/batching.py','sdxl_parallel/batching.py')
with zipfile.ZipFile(out/'sdxl_research_modules.zip') as z:assert z.testzip() is None

def file(name,label=None):
 return '<a href="'+(src/name).as_uri()+'">'+html.escape(label or name)+'</a>'

body=f'''<h1>可替换实验模块预备版</h1><p>2026-10-22 · 方案未定，默认设置保留 · 47 项 CPU 测试通过</p>
<p class="box">已实现并接入训练循环，配置可直接替换。没有增加训练框架、插件注册系统或自动组合试验。以下候选均未开展完整 SDXL GPU 训练，不代表选定方案或效果提升。</p>
<h2>可替换的四类模块</h2>
<table><tr><th>模块</th><th>默认</th><th>候选与用途</th></tr>
<tr><td>{file('stage_policy.py','阶段及损失分配')}</td><td>G 循环 2/3/4；fake 循环 1/2/3/4；GAN/DMD 分配沿用 variant</td><td>指定 G 阶段及各阶段 DMD/GAN；fake 可覆盖学生阶段，或跟随当前 G 阶段。也保留 final-only 诊断选项，明确其覆盖限制。</td></tr>
<tr><td>{file('gan_losses.py','GAN 目标函数')}</td><td>hinge D + −D(fake) G</td><td>logistic D + non-saturating G；成对切换，R1 保持原设置。</td></tr>
<tr><td>{file('pixel.py','判别器')}</td><td>冻结 DINO + RGB 分支</td><td>纯 RGB 分支，不加载 DINO；用于区分预训练特征与局部像素判别的贡献。</td></tr>
<tr><td>{file('image_views.py','GAN 图像视图与增强')}</td><td>局部 latent 解码；仅 D 随机翻转</td><td>整图解码后取同一位置、D/G 同时翻转、关闭翻转。用于检查局部 VAE 解码偏差及增强影响。</td></tr></table>
<p>采样点、PCM 边界、DMD 加噪上界、权重、学习率、batch、造样本 batch 等继续沿用现有参数。未写 SD3/DiT 的空壳适配器；它们的噪声参数化与条件接口需要另行实现。</p>
<h2>8 份独立配置</h2>
<p>都基于相同 uniform_joint4：999/749/499/249，batch64，G160，5:1。默认只是生成命令，必须显式 --execute 才运行。</p>
<table><tr><th>配置</th><th>相对 baseline 的变化</th></tr>'''
examples=[('baseline','无变化，作为当前控制组'),('rgb_only','discriminator=rgb'),
 ('logistic_gan','gan_loss=logistic'),('symmetric_flip','gan_augment=both'),
 ('full_decode_crop','gan_view=full'),('critic_student_cycle','fake_policy=student_cycle'),
 ('critic_match_student','fake_policy=match_student'),
 ('last_step_gan_only','只更新 G 第4步，DMD关闭、GAN开启；仍保留 PCM。这是另一个配方，不是单因素对照。')]
for name,description in examples:
 body+=f'<tr><td>{file("configs/modules/"+name+".json",name)}</td><td>{html.escape(description)}</td></tr>'
body+='''</table><h2>使用与验证</h2>
<pre>python sdxl_research/launch.py --root ROOT --gpus 0,1 \
  --config sdxl_research/configs/modules/rgb_only.json --short</pre>
<p>这条命令只预览。上轮独立审计发现的 --short+resume 问题已修复：出现该组合会明确拒绝。解析后的模块选项写入 config 和完整存档；恢复时检查一致性，防止无意中换实验分支。旧版没有模块元数据的完整状态不会直接接入此版，旧权重仍可用于评测。</p>
<ul><li>47 项 CPU 测试全部通过，包括原有 38 项和新增 9 项模块检查。收尾修复了自定义阶段下 GAN 梯度诊断的计数，现按实际 GAN 更新次数采样，无额外存档计数器。</li>
<li>默认阶段分配、hinge 数值/梯度、局部裁剪数值/梯度与原路径一致；原 hybrid 判别器相关测试继续通过。</li>
<li>logistic 梯度方向和数值有限性、RGB-only 不加载 DINO及其 D/G 梯度、增强分支和整图裁剪位置均已验证。</li>
<li>服务器验证放在独立目录，CUDA 禁用、nice19、低 CPU 线程数；没有启动训练或替换既有远程实验代码。</li></ul>
<h2>使用边界</h2><ul>
<li>整图解码尽管逐张执行并做梯度检查点，仍可能明显增加 GPU 显存；CPU 小样本验证不代表 A100/4090 上能跑。</li>
<li>GAN 损失与判别器更换会改变尺度，同一个 GAN 系数不保证相同梯度比例；应先看损失、梯度和样图。</li>
<li>critic 跟随当前阶段仍是共享 fake 的交替拟合，不是每阶段独立且精确的条件 score。</li>
<li>GAN D 仍统一使用四步最终负样本。joint34 的第三步 GAN 属于额外实验假设。</li>
<li>last_step_gan_only 为保持循环简单，仍保留 fake 更新，因此不是节省全部 fake 算力的专门纯 GAN 实现；PCM 也仍开启。pcm_weight=0 当前只去除其梯度贡献，不省去计算。</li>
<li>先做单因素对照，再组合；不同 G 阶段循环必须同时报告各阶段更新次数和时间，不能只看总 G 次数。</li></ul>'''
body+=f'<p>{file("MODULES.txt","完整参数与扩展说明")} · <a href="sdxl_research_modules.zip">模块预备版代码包</a> · <a href="evidence/cpu_tests.log">47 项测试日志</a> · <a href="evidence/validation.json">验证记录与源码哈希</a></p>'
doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>可替换实验模块预备版</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.75 system-ui,sans-serif;color:#25354b;background:#f8fafc}h1{font-size:30px}h2{font-size:23px;margin-top:30px;color:#205777}table{width:100%;border-collapse:collapse;background:#fff}th,td{padding:13px;border:1px solid #d4dde9;text-align:left;vertical-align:top}a{color:#1268a2}.box{padding:16px;background:#e6f1e9;border-left:5px solid #438667}pre{white-space:pre-wrap;padding:15px;background:#e9eef4}td,p{overflow-wrap:anywhere}</style>'+body+'</html>'
(out/'report.html').write_text(doc,encoding='utf-8')
for target in re.findall(r'href="([^"]+)"',doc):
 if not target.startswith(('file:','http')):assert (out/target).exists(),target
print(out/'report.html')
