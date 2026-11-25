from pathlib import Path
import json,html
root=Path(__file__).resolve().parents[1]
out=root/'output/sdxl-crossdistill/future_implementation_plan_20261025';out.mkdir(parents=True,exist_ok=True)
scores=json.loads((root/'output/sdxl-crossdistill/coco_uniform_g160_20261025/combined_scores.json').read_text())
page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SDXL 分阶段蒸馏：后续实现与实验方案</title>
<style>body{max-width:1120px;margin:36px auto;padding:0 24px;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fa;color:#183047}h1{font-size:29px}h2{font-size:22px;margin-top:0}section{background:#fff;border:1px solid #dce3ed;border-radius:12px;margin:22px 0;padding:24px}table{width:100%;border-collapse:collapse}th,td{border-bottom:1px solid #e0e6ee;padding:10px;text-align:left;vertical-align:top}.note{color:#5a6d82}.status{background:#fff4dd;border-left:5px solid #b87616;padding:16px}.good{background:#eaf3ff;padding:14px}code{background:#eff3f7;padding:2px 5px;border-radius:3px}a{color:#1764af}li{margin:5px 0}</style>
<h1>SDXL 分阶段蒸馏：后续实现与实验方案</h1><p class="note">2026-10-25 · 静态方案，未启动新实验</p>
<div class="status"><b>当前状态：按用户要求停止。</b>本组训练、等待队列、自动评测和后台跟进均已取消；不自动恢复。已有完成的 G160 权重、COCO10K 分数和日志保留。本次新 joint34 尝试未完成 G160，不计入实验结论。</div>

<section><h2>1. 现有证据与真正缺少的对照</h2>
<p>均匀四点 <b>999/749/499/249、末步 GAN+DMD</b> 已完成完整 COCO10K：<b>FID 23.4191，Patch FID 37.8199</b>。旧采样点的末步联合组为 24.2949 / 48.1145。这个改善同时混合了采样点、PCM 边界及 fake 阶段策略的变化，不能单独归因于 GAN。</p>
<p>官方 DMD2 权重在同协议下为 18.4542 / 20.4367，继续保留为外部质量参照。我们的 G160 短训练与其训练预算不同，当前重点是验证小预算下可重复的改进，再逐步扩大训练。</p>
<table><tr><th>组别</th><th>共同基础</th><th>GAN 生效阶段</th><th>回答的问题</th></tr>
<tr><td>A：均匀纯 DMD</td><td rowspan="3">第1阶段 PCM；学生监督轮换第2/3/4步，均含 DMD；同一原始 SDXL 初始化、训练数据、seed、G160 和运行配置</td><td>无</td><td>本项目 PCM+DMD 基线；不是官方 DMD2 复现</td></tr>
<tr><td>B：均匀末步联合</td><td>第4步</td><td><b>B 对 A</b>：加入末步 GAN 是否有收益？</td></tr>
<tr><td>C：均匀第3、4步联合</td><td>第3、4步</td><td><b>C 对 B</b>：再增加第3步 GAN 是否有收益？</td></tr></table>
<p class="good"><b>建议顺序：A → B → C。</b>先补最关键的无 GAN 对照。历史 B 可以用于快速筛选；若要做严格归因，A/B/C 应在同一个固定运行配置下重新从 G0 训练，避免把历史批量调优过程混进对照。</p>
<p>省算力版本先补 A、C，复用历史 B，仅形成探索性结论。严格三组版本则统一重跑 B。两种版本应明确区分，不能把历史 B 描述成完全匹配的新对照。</p></section>

<section><h2>2. 第一轮固定的训练设置</h2>
<table><tr><th>项目</th><th>固定值／规则</th></tr>
<tr><td>模型与数据</td><td>原始 SDXL Base；已有同一份 10K LAION 衍生 latent/文本条件；训练文件 hash 固定。每组独立初始化学生、fake score 网络及需要的判别器。</td></tr>
<tr><td>采样与 PCM</td><td>999/749/499/249；PCM 边界749、权重10；保留当前老师两小步及 stop-gradient 学生目标的实现，不同时引入 EMA 或其他 PCM 变体。</td></tr>
<tr><td>DMD</td><td>第2/3/4步；再加噪上限919；fake score 网络按1/2/3/4轮换，fake:G=5:1。</td></tr>
<tr><td>学习率与随机性</td><td>学生/fake 1e-6，判别器1e-4；teacher CFG8、student CFG1；seed=20261017；相同每样本随机输入和顺序。</td></tr>
<tr><td>训练量</td><td>全局 batch64；G160、fake800。B/C 的 D 更新800次、R1生效50次；A不运行 GAN 判别器。G160不是160个数据 epoch。</td></tr>
<tr><td>GAN</td><td>hinge；系数0.00527422968；现有 DINO+RGB 混合判别器；同一局部 latent 解码与 D-only 翻转。</td></tr>
<tr><td>保存</td><td>G40/80/120/160 保存学生权重；最终保存可续训的模型、优化器、计数器及 RNG。评测独立目录保留特征块与完整性标记。</td></tr></table>
<p>B 的 GAN 学生更新为53次，C为106次。第一轮保留相同系数，检验“新增第三步监督”的整体效果；若 C 更好，再补一个 C 的 GAN 系数减半组，区分阶段位置与增加 GAN 总作用次数的影响。系数乘更新次数相等，并不意味着梯度贡献相等。</p>
<p><b>第三步 GAN 的对象：</b>第三步给出的 clean 预测，不是带噪声的下一步输入。第一轮判别器仍按现有实现使用最终四步样本训练；若后续要改成第三/四步混合负样本，应作为单独消融，避免同时改变 G 和 D 两端。</p></section>

<section><h2>3. 代码实现：保留现有模块，收拢运行入口</h2>
<table><tr><th>部分</th><th>已有能力</th><th>后续最小改动</th></tr>
<tr><td><a href="../../../sdxl_research/stage_policy.py">阶段路由</a></td><td>A/B/C 的 DMD/GAN 阶段可配置</td><td>复用现有路由；配置差异应能一眼看出只改了目标阶段。</td></tr>
<tr><td><a href="../../../sdxl_research/gan_losses.py">GAN loss</a>、<a href="../../../sdxl_research/pixel.py">判别器</a>、<a href="../../../sdxl_research/image_views.py">图像视图</a></td><td>hinge/logistic、hybrid/RGB、tile/full 已可替换</td><td>主矩阵固定当前实现。纯 RGB PixelGAN、整图解码只在有收益后各做一项消融。</td></tr>
<tr><td><a href="../../../sdxl_research/launch.py">启动与配置</a></td><td>配置生成命令、GPU 空闲检查</td><td>方法配置与运行配置分开；启动时保存一份展开后的最终配置。将已有三个配置显式补齐经过验证的内存选项，避免依赖省略字段的默认值。</td></tr>
<tr><td>资源预检与退出</td><td>已有各次任务的临时监测脚本</td><td>收拢为一个简短入口：GPU进程、主机内存、祖先cgroup限额检查；只遍历自己的子进程。用户停止后取消整条后续流程。</td></tr>
<tr><td>训练诊断</td><td>已有 raw loss、学生范数、D真假分数、RGB/DINO分支与R1日志</td><td>按阶段统计 raw/weighted loss与梯度；只有确有疑点时补稀疏梯度探针。</td></tr>
<tr><td><a href="../../../sdxl_research/evaluation_core.py">COCO评测</a></td><td>固定64张特征块、断点跳过、完整索引校验</td><td>继续复用。每个权重与协议独立manifest；验证权重、数据、种子和预处理后，才复用缓存。</td></tr></table>
<p>保留现有 PyTorch/FSDP2 训练流程即可。未来三组统一从 G0 使用固定的有效 batch 与运行配置，<b>不再为了复刻历史调参过程而强行重演 G2 的批量切换和大存档交接</b>。如果必须改运行布局，三组共同修改并明确记录。</p></section>

<section><h2>4. 资源策略：空闲 GPU 不是唯一条件</h2>
<p>此次4090尝试的教训是：物理内存仍有大量空闲，但父级 <code>user.slice</code> 软限为100 GiB、硬限112 GiB。每个用户自身无限额，也不代表不受祖先限制。仅查询 <code>nvidia-smi</code> 与 <code>free</code> 不足以判断能否训练。</p>
<table><tr><th>候选运行配置</th><th>状态</th><th>使用条件</th></tr>
<tr><td>原服务器双 A100 标称40GB；每卡16，累积2；fake与学生优化器 offload</td><td>参考组已实测到G160</td><td>优先候选；先核验实际CPU内存预算和两张空闲卡，预留运行峰值及完整存档余量。</td></tr>
<tr><td>四张 RTX4090 标称48GB；每卡8，累积2；模型与优化器留GPU</td><td><b>仅准备过配置，尚未GPU实测</b></td><td>作为备选，需单独验证初始化、Adam.step、GAN反向、R1和保存峰值。不能宣称已经可用。</td></tr>
<tr><td>普通24GB 4090方案</td><td>本轮未验证</td><td>需另做容量与吞吐测试；不直接套用48GB卡配置。</td></tr></table>
<p>下次先做一次正式参数下的4次学生更新短测：覆盖第三/第四步 GAN、首次 Adam.step、一次 R1、评估和最终存档。资源预检需检查 cgroup v1/v2 的所有有效祖先限制，并记录 anon/shmem、软限事件和内存压力；不能只看物理空闲量。</p>
<p>通过后选定一个运行配置完成主矩阵。资源不足时停止或等待明确可用时段；不修改服务器共享限额、不与未知任务拼显存。本方案不自动创建新等待队列。</p></section>

<section><h2>5. 评测与诊断怎样支持结论</h2>
<p><b>主评测：</b>各组 G160 做同一份完整 COCO2014 10K，FID 与 Patch FID 为主；相同提示词、seed、分辨率、VAE、Inception、裁剪及缩放。已完成的 SDXL50步和官方 DMD2 结果在协议不变时直接复用。小样本 FID 不与完整10K数值混排。</p>
<p><b>过程监控：</b>沿用固定样图，按G40/80/120/160比较整图与局部；纹理观察优先于单独看CLIP。固定128张COCO样图可用于复核，另设少量明确标注为诊断的毛发、织物、植被、细线与人脸提示词。不要按观感挑最佳图。</p>
<p>Patch FID 的中心299裁剪也受构图和语义内容影响，不是纯粹的纹理真实性指标。展示整图、固定中心裁剪，以及可追溯标注的主体局部；避免把更锐、更高频或局部解码差异直接当作更好。</p>
<p><b>梯度诊断：</b>先在固定小批样本上测加权 GAN 与 DMD 对同一 clean latent 的梯度RMS和余弦，判断末端方向是否冲突。PCM处于不同的latent节点，不能直接套同一latent夹角；必要时在固定学生参数子集测三项梯度，并明确它不是全参数证据。现有接口仅有GAN latent RMS，DMD与夹角属于待补内容。损失标量比例不能代替梯度贡献。</p>
<p>最终同时报告质量和实际GPU小时。先按相同G更新数比较；A没有判别器计算，耗时更短是方法成本的一部分。单个训练seed上的10K图像不能替代多个独立训练seed；只有明显正信号后，再对获胜组与A做成对重复。</p></section>

<section><h2>6. 预算与推进条件</h2>
<p>参考组 G3–160 的实际续训段约8.67小时。若能稳定使用原双卡且吞吐接近参考，三组依次训练可先按<b>约24–33小时墙钟时间（双卡合计约48–66 GPU·小时）</b>规划，再加约2–4小时的评测与检查。这个估算不含排队，也不是完成承诺。</p>
<p>省算力版本只新增A/C，预算更低；严格版本统一跑A/B/C，优先安排完整、可持续的空闲时段，争取在2–3天内完成。若资源不能保证，就不同时展开新判别器、整图GAN、GAN权重网格或SD3-Medium迁移。</p>
<ol><li>先核对资源与一个固定运行配置，通过4G最重路径短测。</li><li>完成A，确认均匀四点 PCM+DMD 的真实基线。</li><li>在同配置下完成B；若B无稳定收益，先定位纹理/梯度问题，不扩大矩阵。</li><li>若B有收益，再完成C；若C更好，补减半GAN系数以区分次数与位置作用。</li><li>对有希望的组做成对重复，再决定扩大训练数据、步数或迁移SD3-Medium。</li></ol>
<p class="good"><b>下一次需要确定的只是：</b>选择省算力探索版还是统一重跑三组的严格版，并确认可持续使用的运行环境。现在仅整理方案，代码主干与训练超参数未按此计划修改，也没有启动这些实验。</p></section>
<p><a href="../coco_uniform_g160_20261025/report.html">已完成七组COCO10K结果</a> · <a href="../joint34_uniform_a100_20261025/report.html">本次已取消任务记录</a></p></html>'''
(out/'report.html').write_text(page,encoding='utf-8')
print(out/'report.html')
