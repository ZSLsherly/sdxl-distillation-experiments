from pathlib import Path
import json, hashlib, statistics, html

root = Path('output/sdxl-crossdistill/dmd2_matched_20261119')
out = root / 'audit_official_20261121'
out.mkdir(exist_ok=True)
runs = root / 'final_results_20261121'
rows = [json.loads(x) for x in (runs/'rank0.jsonl').read_text(encoding='utf-8').splitlines()]
gs = [r for r in rows if r['student_updated']]
assert len(rows)==800 and len(gs)==160
assert [r['iteration'] for r in gs]==list(range(0,800,5))
stats = {'G':len(gs), 'F':len(rows), 'depth_G':[sum(r['depth']==d for r in gs) for d in range(4)],
         'depth_F':[sum(r['depth']==d for r in rows) for d in range(4)],
         'G_clip_gt1':sum(r['student_grad']>1 for r in gs),
         'G_clip_gt10':sum(r['student_grad']>10 for r in gs),
         'F_clip_gt1':sum(r['fake_grad']>1 for r in rows),
         'max_G_grad':max(r['student_grad'] for r in gs),
         'max_F_grad':max(r['fake_grad'] for r in rows), 'windows':[]}
for lo,hi in [(1,20),(21,40),(41,80),(81,120),(121,140),(141,160)]:
    g=[r for r in gs if lo<=r['G']<=hi]; f=[r for r in rows if lo<=r['G']<=hi]
    stats['windows'].append(dict(G=f'{lo}-{hi}', student_dm=statistics.mean(r['student_dm'] for r in g),
        student_gan=statistics.mean(r['student_gan'] for r in g),
        fake_mse=statistics.mean(r['fake_mse'] for r in f),
        fake_gan=statistics.mean(r['fake_gan'] for r in f)))
stats['depth_fake_mse'] = []
for d in range(4):
    stats['depth_fake_mse'].append({'depth':d,
        'G1_20':statistics.mean(r['fake_mse'] for r in rows if r['depth']==d and r['G']<=20),
        'G141_160':statistics.mean(r['fake_mse'] for r in rows if r['depth']==d and r['G']>=141),
        'G_clip_gt1':sum(r['student_grad']>1 for r in gs if r['depth']==d)})
for rank in range(4):
    rr=[json.loads(x) for x in (runs/f'rank{rank}.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rr)==800 and [(r['G'],r['F'],r['depth']) for r in rr]==[(r['G'],r['F'],r['depth']) for r in rows]
files=[root/'code/train_matched.py',*[root/'code/vendor/main'/n for n in ['sd_guidance.py','sd_unified_model.py','sd_unet_forward.py','utils.py']],runs/'config.json',runs/'rank0.jsonl']
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(out/'audit_evidence.json').write_text(json.dumps({'upstream_commit':'8d8fa55633d47cfb81bbc7a892e7248f9518763f','statistics':stats,'sha256':hashes},ensure_ascii=False,indent=2),encoding='utf-8')
report='''DMD2 G160 与 A 差距：官方代码、实际配置与日志只读审查
核查日期：2026-11-21（北京时间）
操作范围：未重跑训练、未修改训练代码或超参。只读核验官方仓库、本地副本、服务器源码哈希及已完成日志；本文件和统计是新增审查产物。

一、结论
不能只归结为“短训练没收敛”。这次是“DMD2目标和原生GAN + A的优化器配方 + FSDP2混合精度移植”，并非官方超参复现。已确认学习率、优化器动量、裁剪和预热差异，且发现移植改变了官方特意保留的FP32 fake-score计算。GAN损失符号、系数、G/F更新次数和先后顺序未发现漏项。这些差异与有限的每阶段监督共同存在，现有一条训练轨迹不能定量分摊各自影响。
优先怀疑优化配方和精度改变，之后再判断训练长度；不建议把当前配方直接加时当作唯一补救。

二、结果口径
A：FID 25.617357；中心裁剪Patch FID 33.376656；CLIP 31.674725。
本次DMD2：FID 27.421340；Patch FID 44.179737；CLIP 31.634403。
差值：FID +1.803983；Patch FID +10.803081；CLIP -0.040321。
FID及Patch FID使用同一COCO2014固定10K、相同真实特征；CLIP是历史固定1024提示词、OpenAI ViT-B/32余弦×100。CLIP配对差值bootstrap区间[-0.15046, 0.07021]，语义差距不明确。主要差距在整体与局部视觉分布；Patch FID不是纯粹纹理指标。
本次完整评测已经改用官方随机再加噪四步采样，不能再用旧DDIM预览的彩色伪影解释这里的分数。旧训练中间预览不是官方采样，不能用于断言G40后崩溃。官方发布模型的论文分数不能直接当作本10K短训应达到的数值。

三、关键参数对照（官方SDXL全参数四步，不混用LoRA或SD1.5设置）
项目                         官方                         本次
学生学习率                   5e-7                         1e-6（2倍）
fake和GAN头学习率            5e-7                         5e-6（10倍）
AdamW betas                  (0.9,0.999)                  (0,0.999)
weight decay                 0.01                         0
梯度裁剪阈值                 10                           1
预热                         constant_with_warmup          无
有效batch                    128                          64=4卡×16
fake:student次数             5:1                          5:1
G GAN系数                    0.005                        0.005
fake GAN系数                 0.01                         0.01
教师CFG/fake CFG             8/1                          8/1
学生DM加噪                   20–980（含端点）              20–979
fake回归和GAN加噪            0–999                        0–999

学习率和Adam设置沿用A是当时“公共设置对齐A”的有意选择，不应伪称抄错官方参数；但这不保证DMD2适用。fake每次更新已是官方10倍学习率，同时仍更新5次；beta1=0、没有预热也改变早期动态。不能把5次更新等同于原论文的完整稳定化配置。
预热细节：官方默认warmup_step=500，但固定依赖accelerate 0.23的scheduler在split_batches=False时按进程数推进，且G scheduler每个外循环都step。64卡约8个外循环完成预热，并不是500次学生更新；未来单独恢复预热必须按有效更新数明确实现。
裁剪实测：160次学生更新中44次超过1，包含全部42次t=999更新；只有1次超过10。fake 800次没有超过1。这说明本次裁剪确实改变了学生更新，不是无效配置。Adam下梯度整体缩放与参数变化并非线性对应，不据此宣称更新量缩小固定倍数。

四、移植引入的精度偏差：需承认是额外变量
官方sd_guidance.py的compute_distribution_matching_loss在no_grad中不启用autocast，fake UNet为FP32；原注释明确说明使用full precision。官方FSDP配置mixed_precision=no，生成前缀sample_backward也没有autocast。教师本身为BF16；带梯度生成、fake训练和GAN特征通过显式autocast使用混合精度，GAN头FP32。
本次train_matched.py第85行起对学生、fake和教师使用MixedPrecisionPolicy(param_dtype=bfloat16, cast_forward_inputs=True)。FSDP2在模块前向时将参数和输入转换成BF16，即使外层no_grad且未启用autocast。因此fake score和无梯度生成前缀也变为BF16；之后把输出转double不能恢复已丢失的精度。模块整体BF16也不等同于算子级autocast。
这不是只改变分片/性能：它改变用于计算teacher−fake方向的网络数值。代码路径可确认，但没有同权重FP32/BF16成对误差实验，不能宣称已证明它造成全部FID差距。之前若称“只改运行架构、不改计算”，表述不准确。
后续若做修正复现，应保留student/fake官方FP32参数前向语义，使用官方局部autocast；teacher继续BF16。显存和速度需重新短测，不在本次执行。

五、GAN与fake核查
G目标 = loss_dm + 0.005×gen_cls_loss。
fake+判别头目标 = epsilon回归MSE + 0.01×guidance_cls_loss。
学生GAN为softplus(-D(fake))；判别器为softplus(D(fake))+softplus(-D(real))。
原生判别器是fake UNet瓶颈特征上的分类头，参数与fake一起优化，并在0–999上给输入加噪；不是过去独立DINO+RGB的Pixel GAN。
学生更新时冻结fake参数但保留对生成latent的梯度；fake更新时生成latent已detach。teacher不进入优化器。checkpoint重算期间继续冻结fake，避免G反向写入fake；源代码未发现GAN梯度被显式切断、符号写反或loss少除/多除world size。尚未做全SDXL梯度数值等价验证，不把静态检查说成全部数值已证明等价。
真实顺序与官方一致：外循环0更新G再更新F；循环1–4只更新F；循环5再G+F。不是“先预训练5次F再第一次G”，这不是本次独有错误。800个循环中G在0/5/.../795更新，共G160/F800。GAN从开头启用符合官方SDXL四步脚本，不应借用其他模型的分阶段开GAN流程来判错。

六、同G160不等于相同监督量
本次直接监督生成位置999/749/499/249分别42/37/50/31次；fake位置分别223/189/216/172次。
A每次计算首步PCM、随机一个中段DM及末步DM；末步有160次直接监督。DMD2每次只选一个位置计算clean-DM和GAN；末步31次。四步共享学生UNet参数，因此不是末步参数仅更新31次，而是末步直接监督仅31次。
A的轨迹约束可能帮助极短预算训练，这属于合理解释，尚未被单变量实验证实。
官方README全参数四步发布权重为checkpoint_model_019000，约19K外循环；按官方每5循环一次G是约3800次G，不是19000次G。官方batch128，本次G160/batch64的学生样本量约为该发布预算的2.1%。论文F.4报20K iterations；启动脚本train_iters=100000000只是上限，不是作者实际训练量。

七、loss说明了什么
G1–20的fake回归MSE均值约0.0271，G141–160约0.1442；按四个生成位置分开，早晚MSE也都上升。学生DM均值约0.6392降至0.2683；G GAN raw loss约5附近，判别器raw loss较低。确实值得怀疑fake跟踪压力或判别器失衡，不能以DM下降认定收敛。
但fake训练样本本身随学生变化，MSE不是固定分布的留出误差；GAN loss也不能直接代替梯度强弱/过拟合证据。没有保存最终fake权重或历史留出诊断，无法现在补出旧fake的留出表现；没有官方采样的中间完整FID，无法确定最终是否优于训练中期。不要把训练MSE上升说成已证实fake跟不上，也不能说只要加时一定变好。

八、次要差异与数据检查
数据是与A一致的官方DMD2 latent shard衍生10K，已含VAE scaling，不应再缩放。官方四步路径prepare_denoising_data实际使用real_dataset的条件；虽然配置另加载LAION captions，不能简单声称这条路径每次使用全部3M caption。官方real LMDB为500K，本次10K在800个fake更新中访问51200张次，约5.12遍；判别器可能更易过拟合，但无留出真假评分无法证实。
FP16源SDXL权重转换为FP32训练参数、缓存文本精度、库版本、全局batch与分片方式亦不同。20–979与20–980的一点端点差异存在，但不应优先拿它解释很大的Patch FID差距。

九、下一步建议（仅建议，未执行）
先把此结果命名为“A优化器配方下的DMD2短训对照”，保留有效分数，不把它当原生DMD2复现。若继续，应优先对齐官方FP32 fake-score路径、双5e-7、betas(.9,.999)、clip10及明确的短预热，保留本项目10K/batch64/G160以控制成本；同时记录固定留出fake误差、真假评分和阶段拆分曲线。一次性改这些可验证“官方优化配方是否整体更好”，不能分离单个因素的因果贡献；要归因需再做单变量。不要先花钱盲目拉长现有配方。

十、证据与核查范围
官方commit：8d8fa55633d47cfb81bbc7a892e7248f9518763f。
实际服务器：172.18.132.41。
实际训练：/data1/shenglin.zeng/dmd2_matched_10k_20261119/runs/formal_g160_restart_20261120。
本地：output/sdxl-crossdistill/dmd2_matched_20261119/final_results_20261121。
本轮只读远端sha256sum与本地代码、配置、rank0日志一致；详细哈希及四rank计数复核见audit_evidence.json。
完整COCO报告：../coco10k_official_20261121/report.html。
官方脚本：https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/experiments/sdxl/sdxl_cond999_8node_lr5e-7_denoising4step_diffusion1000_gan5e-3_guidance8_noinit_noode_backsim_scratch.sh
官方训练器：https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/train_sd.py
官方guidance：https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/main/sd_guidance.py
官方模型与训练时长：https://github.com/tianweiy/DMD2/blob/8d8fa55633d47cfb81bbc7a892e7248f9518763f/experiments/sdxl/README.md
论文：https://arxiv.org/html/2405.14867v2
预热实现：https://github.com/huggingface/accelerate/blob/v0.23.0/src/accelerate/scheduler.py
'''
(out/'审查报告.txt').write_text(report,encoding='utf-8')
(out/'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>DMD2与A差距审查</title><style>body{max-width:1050px;margin:36px auto;padding:0 24px;background:#f5f7fa;color:#172033;font:16px/1.85 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:white;padding:30px;border-radius:12px}</style><h1>DMD2 与 A：差距原因审查</h1><p>2026-11-21 · 只读核查 · 未重跑</p><pre>'+html.escape(report)+'</pre></html>',encoding='utf-8')
print(json.dumps(stats,ensure_ascii=False,indent=2))
print(out.resolve())
