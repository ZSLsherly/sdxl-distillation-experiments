from pathlib import Path
import json
pages=[]
refs={}
def p(text):return {'type':'p','text':text}
def h(text):return {'type':'h2','text':text}
def small(text):return {'type':'small','text':text}
def t(headers,rows,widths=None):return {'type':'table','headers':headers,'rows':rows,**({'widths':widths} if widths else {})}
def page(title,*blocks,cover=False):pages.append({'title':title,'blocks':list(blocks),'cover':cover})
def ref(n,title,meta,url,code=None):refs[f'{n:02}']={'title':title,'meta':meta,'url':url,**({'code':code} if code else {})}

ref(1,'One-step Diffusion with Distribution Matching Distillation','Yin 等 · 2023-11-30 首发 · CVPR 2024','https://arxiv.org/abs/2311.18828')
ref(2,'Improved Distribution Matching Distillation for Fast Image Synthesis','Yin 等 · 2024-05-23 首发 · NeurIPS 2024','https://arxiv.org/abs/2405.14867','https://github.com/tianweiy/DMD2')
ref(3,'Learning Few-Step Diffusion Models by Trajectory Distribution Matching','2025-03 · ICCV 2025 · TDM','https://arxiv.org/abs/2503.06674')
ref(4,'SenseFlow Scaling Distribution Matching for Flow-based Text-to-Image Distillation','2025-06 首发 · ICLR 2026','https://arxiv.org/abs/2506.00523','https://xingtongge.github.io/SenseFlow-page/')
ref(5,'Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis','2025-07-24 首发 · ICCV 2025 · ADM 与 DMDX','https://arxiv.org/abs/2507.18569')
ref(6,'Phased DMD Few-step Distribution Matching Distillation via Score Matching within Subintervals','2025-10-31 首发 · CVPR 2026','https://arxiv.org/abs/2510.27684')
ref(7,'Distribution Matching Distillation Meets Reinforcement Learning','2025-11-17 首发 · ECCV 2026 · DMDR','https://arxiv.org/abs/2511.13649','https://github.com/vvvvvjdy/dmdr')
ref(8,'Flash-DMD: Towards High-Fidelity Few-Step Image Generation with Efficient Distillation and Joint Reinforcement Learning','2025-11-25 首发 · CVPR 2026','https://arxiv.org/abs/2511.20549')
ref(9,'Decoupled DMD: CFG Augmentation as the Spear, Distribution Matching as the Shield','2025-11-27 首发 · 预印本','https://arxiv.org/abs/2511.22677')
ref(10,'Diversity-Preserved Distribution Matching Distillation for Fast Visual Synthesis','2026-02-03 首发 · ICML 2026 · DP-DMD','https://arxiv.org/abs/2602.03139','https://github.com/Multimedia-Analytics-Laboratory/dpdmd')
ref(11,'Cross-Resolution Distribution Matching for Diffusion Distillation','2026-03-06 首发 · 2026-06-30 v2 · RMD','https://arxiv.org/abs/2603.06136')
ref(12,'Continuous-Time Distribution Matching for Few-Step Diffusion Distillation','2026-05-07 首发 · CDM','https://arxiv.org/abs/2605.06376','https://github.com/byliutao/cdm')
ref(13,'Spectral Amplitude Purification in Distribution Matching for Diffusion Distillation','2026-09-24 首发 · SAP-DMD · 新预印本','https://arxiv.org/abs/2609.29116')
ref(14,'DMAD: Distribution Matching as Adversarial Distillation for Fast Visual Generation','2026-10-01 首发 · DMAD · 新预印本','https://arxiv.org/abs/2610.02188','https://github.com/Yzmblog/DMAD')
ref(15,'From Slow Bidirectional to Fast Autoregressive Video Diffusion Models','2024-12 首发 · CVPR 2025 · CausVid','https://arxiv.org/abs/2412.07772','https://github.com/tianweiy/CausVid')
ref(16,'Self Forcing Bridging the Train-Test Gap in Autoregressive Video Diffusion','2025-06 首发 · Self Forcing','https://arxiv.org/abs/2506.08009','https://github.com/guandeh17/Self-Forcing')
ref(17,'LongLive Real-time Interactive Long Video Generation','2025-09-26 首发 · ICLR 2026','https://arxiv.org/abs/2509.22622','https://github.com/NVlabs/LongLive/')
ref(18,'Causal Forcing','2026-02-02 首发 · ICML 2026 · 采用 v5 对照结果','https://arxiv.org/abs/2602.02214','https://github.com/thu-ml/Causal-Forcing')
ref(19,'Causal Forcing++','2026-05-14 首发 · 2026-10-04 v4','https://arxiv.org/abs/2605.15141','https://github.com/thu-ml/Causal-Forcing')
ref(20,'Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency','Zheng 等 · 2025-10-09 首发 · ICLR 2026 · rCM','https://arxiv.org/abs/2510.08431','https://github.com/NVlabs/rcm')
ref(21,'Causal-rCM A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models','Zheng 等 · 2026-06-24 · 技术报告','https://arxiv.org/abs/2606.25473','https://github.com/NVlabs/rcm')
ref(22,'AnyFlow Any-Step Video Diffusion Model with On-Policy Flow Map Distillation','Gu 等 · 2026-05-13 首发','https://arxiv.org/abs/2605.13724')
ref(23,'Adaptive Video Distillation','2026-03 首发 · CVPR 2026 · AVD','https://arxiv.org/abs/2603.21864','https://adaptive-video-distillation.github.io/')
ref(24,'Score Gradient Matching Distillation for Few-Step Video Diffusion Distillation','2026-05-28 首发 · ICML 2026 · SGMD','https://arxiv.org/abs/2605.30116','https://github.com/ModelTC/LightX2V/blob/main/lightx2v_train/lightx2v_train/trainers/sgmd.py')
ref(25,'Distribution Matching Distillation without Fake Score Network','Kim 等 · 2026-05-19 首发 · FSF-DMD','https://arxiv.org/abs/2605.19256')
ref(26,'Data-Forcing Distillation Restoring Diversity and Fidelity in Few-Step Video Generation','Chen 等 · 2026-06-16 首发 · DFD','https://arxiv.org/abs/2606.18478','https://github.com/csy2077/data-forcing-distillation')
ref(27,'Uncertainty DMD Restoring Diversity in Few-Step Autoregressive Video Distillation','2026-09-10 首发 · 新预印本','https://arxiv.org/abs/2609.11265')
ref(28,'CoDMD','2026-06-20 首发 · Copula 与关系结构正则','https://arxiv.org/abs/2606.21982','https://github.com/JIA-Lab-research/CoDMD')
ref(29,'DyMD','2026-09 首发 · 动态视频分布匹配 · 新预印本','https://arxiv.org/abs/2609.31349')
ref(30,'From Scores to Samples Elastic Forcing for Autoregressive Video Generation','Zhang 等 · 2026-09-28 首发 · MMD 替代路线 · 新预印本','https://arxiv.org/abs/2609.35491')
ref(31,'PDMD Projected Distribution Matching Distillation for Video Diffusion Models','Wang 等 · 2026-09-28 首发 · 新预印本','https://arxiv.org/abs/2609.35768','https://pdmd2026.github.io/')
ref(32,'LongLive-Plug','2026-09-29 首发 · 新预印本','https://arxiv.org/abs/2609.38154')
ref(33,'FastWan Generating a 5-Second Video in 5 Seconds via Sparse Distillation','FastVideo 团队 · 2025-08-04 · 官方工程报告','https://haoailab.com/blogs/fastvideo_post_training/','https://github.com/hao-ai-lab/FastVideo')
ref(34,'FastWan-QAD FastVideo generates a 5-Second Video in 1.8 Seconds on a Single NVIDIA GeForce RTX 5090 via Quantization-Aware Distillation','FastVideo 团队 · 官方博客标注 2026-06-15','https://haoailab.com/blogs/fastwan-qad/')
ref(35,'FastVideo Official Repository and Wan Recipes','截至 2026-10-13 查阅的官方工程文档','https://github.com/hao-ai-lab/FastVideo','https://haoailab.com/FastVideo/cookbook/wan/')
ref(36,'Multi-Bandwidth Distribution Matching Distillation On the Equivalence of Distribution Matching Distillation and Drifting Models','Zhu 等 · 2026-10-07 首发 · MBDMD · 新预印本','https://arxiv.org/abs/2610.10989')
ref(37,'Few-Step Generation via Data-Space Iteration','2026-10-08 首发 · 新预印本','https://arxiv.org/abs/2610.12102')
ref(38,'Optimizing Few-Step Generation with Adaptive Matching Distillation','2026-02-07 首发 · ICML 2026 · AMD','https://arxiv.org/abs/2602.07345')
ref(39,'Seaweed APT2','官方项目页 · 相邻的对抗后训练路线','https://seaweed-apt.com/2')
ref(40,'Salt 与 Self-Consistency Distribution Matching Distillation','2026-04-03 首发 · SC-DMD · 官方仓库标注 ECCV 2026','https://arxiv.org/abs/2604.03118','https://github.com/XingtongGe/Salt')
ref(41,'Self-Forcing++','2025-10-02 首发 · ICLR 2026 · 长视频训练扩展','https://arxiv.org/abs/2510.02283','https://self-forcing-plus-plus.github.io/')
ref(42,'Diff-Instruct A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models','Luo 等 · 2023-05 首发 · NeurIPS 2023','https://papers.neurips.cc/paper_files/paper/2023/hash/f115f619b62833aadc5acb058975b0e6-Abstract-Conference.html')
ref(43,'Score Identity Distillation','2024-04 首发 · SiD · score 蒸馏相邻路线','https://arxiv.org/abs/2404.04057')

page('综述摘要',
 p('**核心判断：DMD 已从一种单步图像蒸馏方法，发展成图像、视频和流式生成中可组合的训练框架。** 截至 2026 年 10 月，关注点已由“能否把几十步压成几步”，转向“少步后能否保住多样性、运动、语义与训练稳定性，以及能否降低整条生成链路的延迟”。'),
 h('最值得把握的五个变化'),
 p('**图像：大型 flow 模型成为主战场。** SenseFlow、Flash-DMD、Decoupled DMD 与 CDM 分别从 fake score 跟踪、噪声阶段分工、CFG 分解和连续时间对齐改善蒸馏。[04][08][09][12]'),
 p('**视频：训练时的历史必须接近推理时的历史。** CausVid 打开双向教师到因果学生的路线，Self Forcing 使用学生自己的生成历史，之后扩展到因果教师、长视频与可复用蒸馏插件。[15][16][18][32]'),
 p('**多样性与运动成为独立优化目标。** DP-DMD、DFD、Uncertainty DMD 等显示，单一质量总分会掩盖同提示不同种子趋同、运动减弱和颜色过饱和。[10][26][27]'),
 p('**fake score 是新一轮算法竞争的中心。** SGMD、FSF-DMD、PDMD 与 DMAD 分别重设目标、消除部分在线拟合、过滤误差或用密度比判别替代 score 拟合。[24][25][31][14]'),
 p('**实际部署走向联合优化。** 少步蒸馏还需结合分辨率调度、低比特计算、注意力实现和快速解码。FastWan-QAD 的低延迟包含这些组合，不能全部归因于 DMD。[11][34]'),
 small('范围：DMD 指 Distribution Matching Distillation。重点为图像生成、文生视频和图生视频推理加速；不讨论 Dynamic Mode Decomposition。报告基于公开论文、正式会议页面、作者项目及官方仓库，不含本地模型复现实验。'),cover=True)

page('01 阅读地图与证据口径',
 h('报告如何组织'),
 t(['阅读目的','建议阅读内容'],[
 ['建立技术框架','第 02 至 03 节：DMD 原理、DMD2 与方法边界'],
 ['跟进图像方向','第 04 至 07 节：大型 flow、训练分工、多样性及量化证据'],
 ['跟进视频方向','第 08 至 13 节：因果生成、长视频、混合蒸馏和新目标'],
 ['准备复现或部署','第 14 至 18 节：系统、评测、实验顺序和研究机会'],
 ['只看最新动态','第 19 节：2026 年 9 至 10 月前沿清单；文末来源可点击']],[108,383]),
 h('时间线的三个阶段'),
 p('**2023 年末至 2024 年：建立基础。** DMD 证明分布匹配可以实现高质量单步图像生成；DMD2 以更稳定的在线 score 学习和对抗监督扩展到高分辨率少步生成。[01][02]'),
 p('更早的 Diff-Instruct 已用沿扩散过程积分的 KL 进行无数据蒸馏，因此分布蒸馏思想并非始于 DMD。SiD 属于相邻的 score identity 与 Fisher 目标路线，不能当作 DMD 的别名。[42][43]'),
 p('**2025 年：扩展到大型模型和视频。** 轨迹分布、flow 模型、偏好对齐，以及因果视频生成逐渐汇合。方法开始改变训练数据流和模型可见的上下文，而不只是替换损失函数。[03][04][07][15][16]'),
 p('**2026 年：修复瓶颈并扩大可用边界。** 多样性、critic 偏差、训练成本、长上下文、任意步数和多模态输出成为重点，9 至 10 月仍持续出现新预印本。[10][21][22][27][31][36]'),
 h('证据强度如何理解'),
 p('正式会议论文具有同行评审记录，但其数字仍是作者在指定配置下的结果；新预印本更接近前沿，但尚需独立复现；官方工程博客更贴近部署，却可能混合内核、模型和解码器变化。本文分别标明这些来源，不把它们合成一个“全球排行榜”。'),
 p('日期优先采用 arXiv 首次提交时间；会议年份和修订日期另列。所有“提高”“更快”都限定到原文对照设置。报告中的研究建议是基于跨论文证据的分析，不是原论文已证明的结论。'))

page('02 DMD 的机制与方法边界',
 p('学生 Gθ 将噪声 z 和条件 c 映射成图像或视频 x。训练时对学生输出重新加噪得到 xₜ，比较当前学生分布与教师分布在不同噪声尺度上的差异。score 指对输入的对数密度梯度，不是一个视觉质量评分。[01][02]'),
 p('x = Gθ(z,c)，xₜ = αₜx + σₜε\nLDM = Eₜ [wₜ KL(pθ,ₜ || pT,ₜ)]\n∇θLDM ≈ E [wₜ (sfake(xₜ,t) − sT(xₜ,t)) ∂xₜ/∂θ]'),
 {'type':'scheme'},
 p('冻结教师提供目标 score；fake score 用学生样本的去噪目标持续学习，以近似当前学生分布的 score。两个方向之差形成学生更新。实践中还涉及条件引导、噪声权重、梯度归一化和截断，因此应区分理想反向 KL 目标与实际估计器。[01][02]'),
 h('与相邻方法的区别'),
 t(['路线','训练的主要对象','与 DMD 的关系'],[
 ['轨迹回归与一致性','ODE 转移或同一轨迹上的端点一致性','提供初始化或补充覆盖；rCM 将其与 DMD 结合'],
 ['Flow matching 与 MeanFlow','瞬时速度或区间平均速度','是建模或蒸馏参数化；可与分布匹配共同使用'],
 ['GAN 与对抗后训练','真实和生成样本的可辨别性','DMD2 增加 GAN；DMAD 用密度比连接两者'],
 ['量化 稀疏注意力 缓存','每次前向的计算量与数据搬运','作用层级不同；能否叠加需要测量']],[100,189,202]),
 small('DMD 一般减少网络调用次数，不必减少参数量。fake-score critic 与 GAN discriminator 是不同角色；“teacher real score”也不意味着真实数据的 score 已精确可得。'))

page('03 从 DMD 到 DMD2',
 h('DMD 先证明单步生成可行'),
 p('原始 DMD 使用教师预生成的噪声与图像配对，并加入感知回归以限制训练漂移。在 ImageNet-64 上，单次前向达到 FID 2.62；SD1.5、512×512、COCO2014-30K 上达到 11.49。它将少步生成的学习对象从“逐步模仿采样器”推进到“匹配最终生成分布”。[01]'),
 h('DMD2 修复在线分布跟踪与训练输入失配'),
 p('DMD2 去掉上述配对回归，采用更频繁的 fake score 更新，典型比例为 fake score 与学生 5:1；加入真实图像的 GAN 监督；多步训练采用学生自身 rollout 的 backward simulation，使训练查询更接近实际推理状态。这三项分别对应跟踪、质量和输入分布问题。[02]'),
 t(['设置','DMD','DMD2','解释'],[
 ['ImageNet-64 1 NFE','FID 2.62','FID 1.28','DMD2 1.28 为延长训练；普通版本 1.51'],
 ['SD1.5 512² COCO30K','FID 11.49','FID 8.35','同一指标仍需核对 CFG 与采样器'],
 ['典型 SDXL 少步','原作主推单步','4 NFE','50 步 CFG 教师常为 100 NFE']],[126,67,72,226]),
 h('不能从这些数字推导什么'),
 p('DMD2 的 8.35 可以优于其 50 步 PNDM、CFG=3 的教师基线 8.59，但原文另一个 200 步 SDE、CFG=2 教师达到 7.21。“超过教师”因此不是脱离采样器和指标的绝对结论。同理，“500 倍”来自特定 EDM 前向次数参照，不能代替 SDXL 端到端计时。[02]'),
 p('取消配对回归不等于完全不使用数据；DMD2 的 GAN 监督仍依赖数据。训练时同时维护学生、冻结教师和可训练 fake score，往往还涉及判别分支，显存和训练费用可能很高。推理时通常只运行学生生成器。'),
 h('为何 DMD2 仍是重要起点'),
 p('它给后续工作提供了可复现的共同结构：固定教师、在线 critic、学生 rollout 与可选对抗正则。2026 年许多新方法可理解为只修改其中一个环节，适合用受控实验判断收益来自哪里。官方实现和 SDXL 权重仍具有基线价值。[02]'))

page('04 图像方向的大型模型与轨迹扩展',
 h('TDM 扩展匹配对象'),
 p('TDM 把匹配从最终输出扩展至生成轨迹分布，并引入 sampling-step conditioning 与受 consistency 启发的替代目标，以支持不同采样步数。它不要求学生逐点重现教师的整条 ODE 轨迹；重点是学生轨迹所诱导的分布。实验覆盖 SD1.5、SDXL、PixArt-α，说明 DMD 正逐渐与轨迹学习融合。[03]'),
 h('SenseFlow 解决大规模 flow 蒸馏的跟踪问题'),
 p('在 SD3.5-Large 8B、FLUX.1-dev 12B 等模型上，学生更新可能快于 fake score 的适应。SenseFlow 通过 IDA 将学生参数信息混入 fake score，并用 ISG 汇聚区间内的教师指导，配合视觉基础模型判别器改善质量。其关键是让指导信号跟得上生成器变化。[04]'),
 p('SD3.5 的 4 NFE Euler 版本 GenEval 为 0.7098，接近 80 NFE 教师的 0.7140；FLUX 版本为 0.6420，低于教师 0.6699。大模型蒸馏的收益因骨干和能力项而异。该文 FID-T 参照教师样本，与参照真实 COCO 分布的 FID 也不是同一指标。[04]'),
 h('ADM 与 DMDX 增强对抗分布匹配'),
 p('ADM 在 real 与 fake score 的 latent prediction 上构造对抗匹配，DMDX 将对抗预训练和 ADM 微调串联，缓解仅靠反向 KL 的质量与覆盖权衡。单步 SDXL 的 MPS 为 11.1978，对照 DMD2 10.6947；论文给出的 A100 总训练成本为 2240 对 3840 GPU 小时，前者包含 ODE 数据收集。[05]'),
 h('从研究设计上如何理解这些工作'),
 p('TDM 改变匹配对象，SenseFlow 改善在线跟踪，DMDX 增加对抗学习结构。这三类改进并不互斥，但直接叠加未必有效：它们会改变教师查询分布、fake score 的学习速度和学生梯度尺度。复现时应逐个验证，再测试组合。'),
 small('名称提醒：2025 年的 ADM/DMDX 与 2026 年 10 月的 DMAD 是不同工作；后者以判别密度比替代辅助 score 拟合，见第 12 节。'))

page('05 图像方向的训练分工与偏好对齐',
 h('Flash-DMD 按噪声阶段分配任务'),
 p('Flash-DMD 让高噪声阶段侧重整体结构的分布匹配，低噪声阶段侧重像素 GAN 的纹理细节，之后再以 latent RL 配合持续蒸馏进行改进。它强调不同噪声阶段不应承担完全相同的学习任务。[08]'),
 p('其 SDXL 4 NFE、COCO10K 的 1000 次迭代版本，ImageReward 为 0.9509，对照 DMD2 0.8748；CLIP 却从 0.3302 略降到 0.3292。“2.1% 训练成本”的定义是 batch×iterations 比例，不是 GPU 小时，也不能写成端到端训练快约 48 倍。[08]'),
 h('Decoupled DMD 拆开分布匹配与 CFG 增强'),
 p('该工作将带 CFG 的实际梯度分成条件分布匹配项和教师条件与无条件差分形成的引导增强项，并给两者不同的重新加噪区间。它提出后者在复杂文生图少步转化中作用很大，前者可承担抑制伪影的正则角色。这是对具体实现的机制分析，不是推翻 DMD 的理想 KL 目标。[09]'),
 h('CDM 将对齐延伸至连续时间'),
 p('CDM 用随机长度的连续时间 schedule 与 off-trajectory latent 对齐，减少固定采样锚点限制。Longcat-Image 4 NFE 的 DPG 为 88.35、HPSv3 为 10.65；对照作者复现的 image-free DMD2 为 87.12 与 8.803，并非完整官方 GAN 版本。无需额外 GAN 或奖励目标不等于无需 fake-score。[12]'),
 h('DMDR 让偏好优化与分布约束共存'),
 p('DMDR 将 reward-tilted 分布匹配、动态引导与后续 RL 相结合，持续保留 DMD 约束以减轻 reward hacking。4 步 SD3.5-Large GenEval 0.72，对照教师 0.71，但 counting 子项 0.73→0.69。偏好和组合能力仍需分别测量。官方仓库的公开训练范围主要是 ImageNet 演示，不能等同于所有大规模文生图配方已经齐全。[07]'),
 small('综合分析：噪声阶段分工、CFG 分解和 RL 引导都在改变学生“优先学什么”。它们对某项分数的提升，不自动意味着分布覆盖、文字渲染或长提示理解同步改善。'))

page('06 多样性与每步计算量',
 h('Phased DMD 在时间区间内匹配'),
 p('Phased DMD 依据 SNR 划分子区间，构造区间内 score 匹配，结合阶段专家容量，改善少步梯度截断下的分布覆盖。在 Qwen-Image 上，4 步 LPIPS 多样性由 0.309 提高至 0.322；DINOv3 余弦相似度却由 0.941 升至 0.958，而此项越低通常表示越多样。两种指标并不一致，应完整呈现。[06]'),
 h('DP-DMD 把首步作为多样性的入口'),
 p('DP-DMD 首步使用教师 target-prediction，例如 v-prediction，后续步再使用 DMD，并在首步输出处停止 DMD 梯度，避免沿首步计算图反传。共享权重仍会更新。其意图是保留早期样本差异，再由后续步骤提高质量；它不增加额外模块，但保留标准 DMD 的 fake score。[10]'),
 p('SD3.5-M、COCO10K、4 NFE：DINO 多样性从 0.210 提高至 0.250，CLIP 多样性从 0.154 到 0.182，VQ-R1 几乎保持 4.690→4.689；教师 DINO 多样性为 0.288。改善显著但尚未完全追平教师，适合做覆盖与质量权衡的受控基线。[10]'),
 h('RMD 在步数之外降低计算量'),
 p('RMD 先低分辨率建结构、后高分辨率补细节，通过 logSNR 校准与噪声重注入处理跨分辨率分布差异。SDXL 的 2+2 步在单 A100 上报告相对教师 33.4 倍加速，相对同论文的全分辨率 4 步基线约 1.67 倍；Wan2.1-14B 的 3+3 步为 25.6 倍及约 1.53 倍。[11]'),
 h('这些工作改变了什么认识'),
 p('“四步”不是一个充分的成本描述：四个高分辨率步骤和两个低分辨率加两个高分辨率步骤，计算量不同。也不能假设每一步都应采用相同目标：最早的结构选择可能决定生成分布是否已经收缩，后续细节优化未必能重新恢复丢失的模式。'),
 small('多样性必须限定为同提示多种子、跨提示覆盖或时间动态中的哪一种；它们相关，但不能相互替代。'))

page('07 图像量化证据与反例',
 p('下表每行只比较原论文中的对照。不同数据集、教师、提示词、CFG 和 FID 参照分布之间没有统一排名关系。'),
 t(['方法与设置','论文结果','必须保留的限制'],[
 ['DMD2\nSD1.5 512² 1 NFE [02]','COCO30K FID 8.35\n50 步教师 8.59','更长 SDE 教师可达 7.21'],
 ['SenseFlow\nSD3.5 4 NFE [04]','GenEval 0.7098\n教师 0.7140','接近教师；FLUX 对照下降更明显'],
 ['DMDX\nSDXL 1 NFE [05]','MPS 11.1978\nDMD2 10.6947','MPS 不等于全能力评估'],
 ['DMDR\nSD3.5-L 4 步 [07]','GenEval 0.72\nDPG 85.33','counting 仍有下降'],
 ['Flash-DMD\nSDXL 4 NFE [08]','ImageReward 0.9509\nDMD2 0.8748','2.1% 为训练样本处理量代理'],
 ['Decoupled DMD\nSDXL 4 步 [09]','FID 18.95→17.80\nCLIP-S 33.14→33.62','HPSv2.1 30.64→30.34'],
 ['DP-DMD\nSD3.5-M 4 NFE [10]','DINO 多样性\n0.210→0.250','教师为 0.288；仍有覆盖缺口'],
 ['SAP-DMD\nPixArt 512² 4 NFE [13]','ImageReward\n0.7945→0.8761','此行 DMD2 对照未使用 GAN'],
 ['DMAD\nImageNet64 1 NFE [14]','基础 FID 1.24\n增强判别器后 1.04','1.04 不能归因于单一损失替换']],[149,145,197]),
 h('怎样阅读“优于教师”'),
 p('教师可能为速度、CFG 或审美做了特定选择；蒸馏可能同时引入真实数据、奖励模型、判别器甚至不同数据分布。学生某项分数更高，说明该训练方案在该评测上更合适，不能据此认定学生继承或超过教师全部知识。'),
 small('表内数字来自对应论文，不是本报告在统一环境中的复测。需要统一排名时，必须重新固定提示集、种子、模型版本、输出尺寸、推理实现和评估脚本。'))

page('08 视频为什么比图像更难',
 h('有两个独立的加速轴'),
 p('第一个轴是**扩散步数**：把每段视频的几十次去噪变成 1 至 4 次。第二个轴是**时间生成顺序**：把整段双向注意力生成改为逐块或逐帧因果生成，让首段更早可见，并复用历史 KV cache。DMD 主要处理前者，CausVid 与 Self Forcing 等把两者结合。[15][16]'),
 h('误差会通过生成历史继续传播'),
 p('图像样本之间通常相互独立；因果视频的下一段以已经生成的历史为条件。如果训练总看真实或教师历史，推理却只能看学生自己的历史，微小错误就可能逐段累积。纹理、主体身份、相机轨迹和动作都可能漂移。少步误差与自回归误差会相互影响。'),
 h('质量与运动也可能相互冲突'),
 p('强烈的质量偏好可能把模型推向更安全、更静止的输出。结果是单帧漂亮、跨帧平滑，但动作不足或不同随机种子趋同。反过来，简单增大运动量也可能只是抖动、形变或镜头乱动；运动幅度必须与自然度、物理一致性和指令服从一起检查。[23][24][27]'),
 t(['视频特有问题','常见处理方式','代表工作'],[
 ['训练与推理历史不同','学生自回归 rollout 后计算分布损失','Self Forcing [16]'],
 ['因果学生与教师不匹配','因果教师或因果 consistency 初始化','Causal Forcing 系列 [18][19]'],
 ['长视频历史持续增长','短窗口 KV 缓存与长 rollout 训练','LongLive [17]'],
 ['多样性与运动被削弱','数据或结构化不确定性监督','DFD / Uncertainty DMD [26][27]'],
 ['少步后每次前向仍贵','低比特 注意力内核 快速解码','FastVideo / QAD [33][34]']],[115,232,144]),
 p('因此，“视频 DMD”不能只按损失函数分类。还要同时记录教师可见上下文、学生因果结构、训练 rollout 长度、chunk 大小、缓存规则和解码方式。这些选择直接决定首帧延迟与长时稳定性。'))

page('09 从 CausVid 到 Causal Forcing',
 h('CausVid 将双向教师蒸馏为因果学生'),
 p('CausVid 以双向扩散教师指导可流式生成的自回归学生，允许学生和教师使用不同时间条件和注意力结构。原始论文以自研 CogVideoX 类教师为基础，352×640、4 步、单 H100 报告 9.4 FPS 与约 1.3 秒首输出延迟。这不是后续 Wan 移植版的同一实验。[15]'),
 h('Self Forcing 使用学生自己的生成历史'),
 p('Self Forcing 在训练中进行自回归 rollout，使学生在自己生成的历史上继续生成，再使用分布级监督。截断梯度和缓存使这种训练可行，方法框架也不局限于一种分布损失。它的核心贡献是对齐训练和推理的数据流，而不只是把 DMD 应用于更多视频帧。[16]'),
 p('Wan2.1-1.3B、832×480、4 步、单 H100：chunk-wise 版本报告 17.0 FPS、0.69 秒首输出、VBench 84.31；frame-wise 为 8.9 FPS、0.45 秒、84.26。更细的生成粒度改善首段响应，却降低吞吐，展示了实时系统必须面对的取舍。[16]'),
 h('Causal Forcing 改善教师与初始化'),
 p('Causal Forcing 先通过 teacher forcing 训练自回归教师，再以因果 ODE 进行学生初始化，最后用 Self Forcing 的 asymmetric DMD 微调。其论文同一协议下 VBench 从 Self Forcing 83.74 提高到 84.04，不能拿前一篇论文的 84.31 与这里直接相减。[18]'),
 h('Causal Forcing++ 降低初始化负担'),
 p('后续版本用因果 consistency distillation 替代离线 ODE 初始化，再接 DMD，减少轨迹存储与训练成本。其第二阶段训练成本报告从 11600 降到 2900 A800 GPU 小时，并免去 1900 GiB 轨迹存储。单 A800 上 2 步 frame-wise 模型为 14.1 FPS、0.27 秒首输出；计时排除 VAE，且首个 latent 帧仍使用 4 步，不能写成“所有帧两步的端到端 0.27 秒”。[19]'))

page('10 视频质量与多样性的修复',
 h('Adaptive Video Distillation'),
 p('AVD 在 DMD 上加入真实视频的自适应回归与时间方差正则，针对颜色过饱和及低动态问题；推理还结合早期半帧率 latent 与插帧。其速度收益不能全部归因于 DMD。时间方差也必须受约束，避免把抖动或噪声当成有效运动。[23]'),
 h('Data-Forcing Distillation'),
 p('DFD 以一定概率让 teacher-score 在加噪参考视频上计算，fake-score 仍估计学生分布，以恢复纹理与多样性。参考池 ViPE-Wild-1M 来自 Wan2.1-14B 生成，不能简称真实拍摄视频。Wan1.3B 的 DINO 多样性 0.190→0.205，教师为 0.301；论文所列 0.906 的 Average VBench 是选定维度均值，不是通常的 Total 90.6 分。[26]'),
 h('Uncertainty DMD'),
 p('该工作在训练与推理中均启用首 chunk 时间扰动和随机 cache 写入。Self Forcing 对照中，VENDI-DINO 2.6→3.3、Dynamic Degree 25.1→42.4，但主体一致性 98.6→97.6；普通额外噪声反而把动态降至 15.9。有效的不确定性必须被转化为合理结构。[27]'),
 h('CoDMD 与 DyMD'),
 p('CoDMD 引入样本间及帧间关系约束，以补强有限容量网络对联合结构的学习。应把“copula 不足”理解为作者对实际优化器与有限估计的诊断；理想的完整 joint score 和联合反向 KL 本来就包含依赖结构，不能概括成“标准 DMD 理论上只匹配边缘分布”。[28]'),
 p('DyMD 面向机器人交互视频，通过 temporal affinity 分配加噪时间，并提高 critic 对难拟合动态的权重。同预算下 R-Bench task adherence 34.6→44.2，而 PAI Quality 均为 78.1，说明执行动作与画面质量是两回事。其任务特定收益不能直接外推到所有文生视频场景。[29]'),
 h('建议同时观察三种现象'),
 p('同提示多种子是否产生不同事件或视角；同一视频内是否有合理运动；跨时间的身份、背景和物理过程是否稳定。静态保真、样本多样性和时间动态至少是三个维度，单个 VBench 总分不能完整覆盖。'))

page('11 长视频与复用能力',
 h('LongLive 将训练长度扩展到真实长时使用'),
 p('LongLive 结合长 rollout、短片段分布监督、KV recache 和局部窗口，缓解持续生成的误差累积，也支持运行中切换提示。论文报告在 Wan1.3B、832×480、单 H100 下达到 20.7 FPS，并展示最长 240 秒生成；训练以 60 秒 rollout、5 秒片段监督为关键配置。[17]'),
 p('这里“支持 240 秒”是已展示的生成范围，不等于所有场景在四分钟内都保持身份或物理一致性。长期指标应覆盖不同时段、提示切换和主体重新出现，不能只取开头几秒。'),
 h('Self-Forcing++ 用短教师纠正长历史'),
 p('Self-Forcing++ 从学生长 rollout 中采样短片段，重新加噪后让短视频教师提供 DMD 纠正，无需长视频教师。它与 LongLive 都说明，短窗口监督仍能作用于长历史下产生的错误；但训练必须让这些错误先真实出现。[41]'),
 h('Elastic Forcing 是从 score 到样本的替代路线'),
 p('Elastic Forcing 在冻结视频表征空间匹配样本 MMD，以 replay 和梯度子采样控制资源，后训练不再在线使用 teacher 与 fake score。它仍依赖已有初始化和离线参考视频，因此属于分布匹配的邻近路线，不是严格 DMD。同协议 VBench 83.80→84.64；已核验三编码器权重对应 84.25，不能混用。[30]'),
 h('LongLive-Plug 将蒸馏转化为可复用适配器'),
 p('LongLive-Plug 将 4 步 DMD2、CFG 调整与长上下文纠错模块化为 LoRA 适配，报告覆盖 Wan2.1-14B、Wan2.2-TI2V-5B、MiniMax-H3 三类家族、54 个下游模型和 8 类任务。其 Wan2.2-TI2V-5B 基础蒸馏示例约需 80 H100 GPU 小时。这指向复用一次蒸馏收益、减少下游重复训练的方向。[32]'),
 h('可复用不等于任意组合无损'),
 p('下游 LoRA 会改变模型输出分布；原蒸馏适配器可能不再校准。需要检查适配器叠加顺序、权重尺度、CFG 条件、步数和专有控制模块。报告中的覆盖规模说明作者进行了较广测试，但尚不能担保未测试模型或任意 LoRA 的兼容性。'),
 p('从落地角度，长视频 DMD 的下一步不只是延长秒数，还包括跨任务复用、交互提示响应、缓存可恢复性，以及在固定显存内稳定运行。这些能力决定模型能否从演示进入长期服务。'))

page('12 fake score 瓶颈与新目标',
 h('为什么在线 critic 会成为瓶颈'),
 p('学生分布持续变化，fake score 要追踪一个移动目标。更新太少会落后，更新太多又增加训练费用；即使充分更新，也可能存在容量与近似误差。这个误差会直接进入学生更新，逐渐积累为过饱和、伪影或覆盖收缩。[02][24][31]'),
 h('SGMD 与 FSF-DMD'),
 p('SGMD 构造 score-gradient matching 与 stop-gradient Fisher 目标，减少内循环依赖。Wan14B 无 GAN 对照的 fake 更新 5→1；论文按前向与反向耗时估算每迭代约快 3 倍。FVD 115.1→100.3、Dynamic Degree 80.56→93.06，但 Quality 85.05→84.77、Semantic 77.46→75.64。FVD 参照教师，不应与真实视频参照混比。[24]'),
 p('FSF-DMD 用 flow-map 生成器端点的 pseudo-velocity 代理 fake velocity，取消独立 fake-score 网络。ImageNet256、2 步实验 FID 为 3.85，对照 DMD2 4.18；但共同初始化已训练 400K 步，不能把后续 4K 步当作总成本。作者将其定位为概念验证，尚不能外推到大规模视频。[25]'),
 h('PDMD 用现有预测过滤 critic 误差'),
 p('PDMD 令 r 为 critic 去噪端点与学生输出之差，将 DMD 更新 d 中平行于 r 的分量去掉：d′ = d − ((d·r)/(r·r))r。理论结论依赖高维集中与弱对齐等假设；实践不增加网络或前向次数。Wan1.3B 4 NFE 在同协议下 VBench 82.70→83.73；MiniMax-H3 音视频实验为 82.76→83.17。[31]'),
 h('DMAD 直接学习密度比'),
 p('DMAD 用共享骨干的真实/学生与教师/学生判别头估计 log-density ratio，在理想判别器条件下恢复分布匹配梯度，从而替代辅助学生 score 拟合。ImageNet64 单步基础版本 FID 1.24，增加 projected discriminator 后为 1.04；这项最新结果仍待独立复现。[14]'),
 small('建议优先做可归因的比较：固定学生更新数、数据、训练时长或 GPU 小时中的一个预算，再记录其他预算。迭代更快与达到同质量所需总训练时间更短，是不同结论。'))

page('13 与 consistency 和 flow map 的融合',
 h('rCM 用互补目标平衡覆盖与细节'),
 p('rCM 把连续时间 consistency 学习扩展到大规模图像与视频模型，并以 score distillation 作为长跳跃正则。其直觉是以 consistency 的覆盖倾向与 DMD 的质量倾向互补，而非要求一种目标独自解决所有问题。工作还提供支持并行训练的 FlashAttention-2 JVP 内核。[20]'),
 p('论文覆盖 Cosmos-Predict2、Wan2.1 等最多 14B 参数模型与 5 秒视频，在 1 至 4 步实现加速。官方仓库同时提醒，sCM 与 DMD 从一开始联合训练在部分模型上会不稳定，更稳妥的配方是分阶段预热再引入 DMD。这是算法与基础设施共同决定可复现性的例子。[20]'),
 h('Causal-rCM 将这种互补带到流式生成'),
 p('Causal-rCM 以 teacher-forcing consistency 初始化，再用 self-forcing DMD 做 on-policy 改善，并提供自定义 mask 的 JVP 内核。2026 年 6 月技术报告给出 2 步因果 Wan1.3B 的 VBench-T2V 84.63，训练使用合成数据。这个数字仅代表其协议，不能与其他论文中 84 分附近的结果直接排次序。[21]'),
 h('AnyFlow 面向可变推理预算'),
 p('AnyFlow 将学习对象从 zₜ→z₀ 的端点映射扩展为任意区间 zₜ→zᵣ 的 flow map，并通过 Flow Map Backward Simulation 进行高效 on-policy 蒸馏。目标是在少步时有效，并在允许更多推理步时继续提升，覆盖双向和因果架构及 1.3B 至 14B 规模。[22]'),
 h('Salt 约束跨步组合与缓存条件'),
 p('Salt 的 SC-DMD 加入一步与两步输运端点的一致性，并在因果版本混合不同步数的历史、对齐缓存条件。其统一评测中 Causal Forcing 4 NFE 的 84.62 提高到 85.08；单独增加一致性损失并非总有利，需要相应 rollout 设计。该分数不能与其他 VBench 协议拼榜。[40]'),
 h('对系统选型的意义'),
 p('固定 4 步模型更容易控制延迟，但未必在改成 8 步时继续变好。若产品提供“极速、标准、精细”多档质量，可变步数的训练目标更合适。若固定预算已满足任务，混合目标的额外训练复杂度则应通过覆盖和语义收益来证明。'),
 p('这些方法都说明，DMD 更像可嵌入的分布约束，而不是与所有 consistency 或 flow 方法互相替代的孤立类别。报告中将 rCM、Causal-rCM 和 AnyFlow 单列为融合路线，以免把组合系统的收益全算作纯 DMD。'))

page('14 从 NFE 到真实端到端速度',
 p('生成时间可以近似拆成：文本编码 + 各次去噪前向 + 视频解码 + 数据搬运及调度。DMD 首先减少去噪次数；当它已从几十步降到三四步，其他部分的占比会迅速上升。'),
 h('FastWan 展示稀疏蒸馏与系统实现的叠加'),
 p('2025 年官方报告中，FastWan2.1-1.3B 在单 H200 生成 5 秒 480p 视频约需 5 秒端到端，去噪仅约 1 秒；单 RTX 4090 对应约 21 秒与 2.8 秒。其训练结合 DMD 与稀疏注意力，并叠加编译和内核优化。[33]'),
 t(['Wan2.1-1.3B 单 H200','去噪耗时','改变的部分'],[
 ['FA2 原始模型','95.21 秒','原始多步参照'],
 ['FA2 + DMD','2.88 秒','压缩采样步数'],
 ['FA3 + DMD','2.14 秒','注意力实现'],
 ['再加编译','1.49 秒','计算调度与融合'],
 ['VSA + DMD + 编译','0.98 秒','稀疏注意力共同优化']],[222,91,178]),
 h('FastWan-QAD 把量化误差纳入蒸馏'),
 p('2026 年官方 QAD 配方先做量化感知微调，再以量化感知 DMD 压到 3 步，使用低比特线性层、低比特稠密注意力、算子融合、编译和 TAEHV 快速解码。5 秒 480p 视频的端到端结果为 RTX 5090 上 1.8 秒，RTX 4090 FP8 版本约 3.4 秒。这不是仅把 DMD2 损失替换后得到的速度。[34]'),
 h('当前工程生态'),
 p('FastVideo 官方仓库已扩展到因果蒸馏、QAD、MiniMax-H3 音视频与 Apple Silicon 等路径；具体精度与内核有硬件限制。2026 年的工程优势越来越依赖“模型权重与推理栈一起交付”，不能只下载权重就期待论文的全部速度。[35]'),
 small('上述计时是作者报告，不包含本地复测。表中的去噪耗时与文中的端到端耗时分别保留；没有把 H200 与 RTX 4090 或 RTX 5090 的数字直接换算成算法优劣。'))

page('15 视频性能对照与口径',
 t(['方法','模型与设置','作者报告的结果'],[
 ['原始 CausVid [15]','自研教师 · 352×640\n4 步 · 单 H100','9.4 FPS\n首输出约 1.3 秒'],
 ['Self Forcing [16]','Wan1.3B · 832×480\nchunk-wise 4 步 · 单 H100','17.0 FPS · 0.69 秒\nVBench 84.31'],
 ['Self Forcing [16]','同骨干 · frame-wise\n4 步 · 单 H100','8.9 FPS · 0.45 秒\nVBench 84.26'],
 ['Causal Forcing [18]','Wan1.3B · 832×480\n4 步 · 单 H100','17 FPS · 0.69 秒\n其协议 VBench 84.04'],
 ['Causal Forcing++ [19]','Wan1.3B · 832×480×81\n2 步 · 单 A800 · 排除 VAE','14.1 FPS · 0.27 秒\n首 latent 帧仍为 4 步'],
 ['LongLive [17]','Wan1.3B · 832×480\n单 H100','20.7 FPS\n展示最长 240 秒'],
 ['RMD [11]','Wan14B · 480p→720p\n3+3 步 · 单 A100','相对教师 25.6 倍\n相对 6 步基线约 1.53 倍'],
 ['FastWan-QAD [34]','Wan1.3B · 5 秒 480p\n3 步 · 单 RTX 5090','端到端 1.8 秒\n含低比特与快速解码']],[116,199,176]),
 h('四类数字不能混用'),
 p('**首输出延迟**决定用户多久看到第一段；**持续 FPS**决定是否跟得上播放；**完整视频端到端时间**反映一次离线任务耗时；**NFE**只反映网络调用次数。实时交互通常同时约束前两者，离线批处理更关注吞吐和成本。'),
 p('CFG 的条件与无条件分支常使一次采样步需要两次前向：50 步教师可能是 100 NFE，而 4 步免 CFG 学生为 4 NFE。25 倍是这组前向次数比，不是 50÷4，也不是必然的端到端提速。'))

page('16 评测应该怎样设计',
 h('图像评测面板'),
 p('同时报告分布保真、提示遵循、偏好和多样性。FID 需要固定真实或教师参照、样本数量和预处理；GenEval 与 DPG 可帮助拆开计数、颜色和空间关系；奖励模型分数应配合盲评；同提示多种子多样性需要避免把坏图或无关输出计作成功。'),
 h('视频评测面板'),
 p('VBench 总分之外保留质量、语义、Dynamic Degree、主体与背景一致性等分项；使用 FVD 时固定视频长度、帧率和样本数。运动提升要排除相机抖动、无意义形变和指令偏离。长视频按多个时间窗口报告质量，检查提示切换后的恢复、动作持续性与主体再出现。'),
 h('训练预算应以相同质量目标对齐'),
 p('至少同时记录学生更新次数、fake score 更新次数、处理样本数、GPU 小时、峰值显存及合成数据生成成本。只报告 batch×iterations 会遗漏每次迭代的网络调用量；只报告每迭代快几倍会遗漏收敛速度。像 Flash-DMD、SGMD 和 Causal Forcing++ 的效率数字，应保留各自的原始定义。[08][24][19]'),
 t(['比较层级','需要固定或披露的条件'],[
 ['生成质量','模型版本 教师 CFG 提示集 种子 数量 输出尺寸 帧数 评估脚本'],
 ['推理速度','GPU 数量与型号 精度 batch 预热 编译 缓存 offload 是否含编码与解码'],
 ['训练效率','有效 batch 更新比 迭代数 数据合成成本 达到目标质量的总 GPU 小时'],
 ['长视频','总时长 chunk 粒度 上下文窗口 cache 刷新 提示切换 时间分段结果']],[108,383]),
 h('一个已核验的可比性实例'),
 p('PDMD 在 Wan1.3B 上使用 944 条增强版 VBench 提示、每条 5 个种子；其 DMD 与 PDMD 对照只差投影，结果相对更容易归因。但这套协议不能直接与使用原始提示集、不同教师或 14B 模型的成绩排名。它的 DMD2 对照还采用额外高质量教师样本做判别正例，也需要一并说明。[31]'))

page('17 优先研究问题与可检验假设',
 h('一 梯度质量能否比增加 critic 更新更划算'),
 p('**假设：**过滤或重加权有偏梯度，可能比增加 fake score 更新次数更节省总训练成本。先对比 DMD2、PDMD 和一种频谱处理方法，固定生成器更新数及数据，再测 GPU 小时、饱和度、覆盖与语义。若只提升视觉分数而持续损失多样性，这个假设就不成立或需要额外约束。[13][31]'),
 h('二 多样性是否主要在最早的生成阶段丢失'),
 p('**假设：**首步或首 chunk 的结构选择决定后续覆盖，后期正则难以补救。比较 DP-DMD 式首步约束、Uncertainty DMD 式结构化扰动，以及把同等约束施加到所有步骤；记录同提示多种子事件差异与质量。控制额外计算后，才能判断收益是否来自“早期”这一位置。[10][27]'),
 h('三 分布覆盖与细节目标如何随时间调度'),
 p('**假设：**先覆盖、后细化的混合训练比固定权重更稳。可以以 rCM、CDM 或 Phased DMD 为基础，检验不同噪声区间的损失配比，关注少步和多步同时可用的能力。不要把训练中某个梯度范数更大等同于该项因果上更重要。[06][12][20]'),
 h('四 低比特误差能否与蒸馏共同校准'),
 p('**假设：**在目标硬件精度下蒸馏优于事后直接量化。用同一学生比较后训练量化、量化感知微调及 QAD，固定解码器和注意力模式，分别观察 denoiser、完整链路时延和质量。FastWan-QAD 给出了可行性，但其完整速度包括多项工程改动。[34]'),
 h('五 长视频能力能否复用于下游模型'),
 p('**假设：**解耦的步数蒸馏与长上下文适配器比逐模型重训更经济。测试多个下游 LoRA、控制任务、CFG 和视频长度，以失败率与所需再训练成本作为结果，而非只展示成功样例。LongLive-Plug 提供了较直接的研究起点。[32]'),
 small('这些是本报告提出的实验方向。它们由已有工作启发，但不是已被全部验证的结论；每个方向都应设定能否定假设的指标。'))

page('18 复现与落地的建议顺序',
 t(['当前目标','建议起点','优先验证'],[
 ['理解 DMD 与建立基线','官方 DMD2 + 一个固定图像骨干 [02]','噪声时间采样 更新比 CFG 归一化与数据流程'],
 ['大型图像 flow 模型','SenseFlow / CDM；配合 DP-DMD 对照 [04][12][10]','图文理解 文字与计数 同提示多种子覆盖'],
 ['视频少步离线生成','官方 rCM 或成熟 FastVideo 配方 [20][35]','质量分项 VAE 开销 显存与总时延'],
 ['低延迟流式视频','Self Forcing / Causal Forcing++ / Causal-rCM [16][19][21]','首输出 chunk 粒度 历史漂移和提示切换'],
 ['训练成本受限','先测 PDMD 等局部修改，再评估新目标 [31][24]','相同质量门槛下总 GPU 小时'],
 ['消费级显卡部署','对应硬件的官方 QAD 权重和实现 [34]','精度及内核兼容 快速解码器造成的质量变化']],[115,207,169]),
 h('第一轮建立可信的速度与质量基线'),
 p('固定一套真实任务提示，并加入计数、文字、快速运动、遮挡和镜头变化等困难样例。图像至少保留同提示多种子对照；视频同时记录首段延迟、持续吞吐与完整时延。先确认自己能接近官方结果，再修改损失。'),
 h('第二轮每次只修改一个关键环节'),
 p('先做低侵入的 critic 投影、首步目标或时间调度变化。完成质量和覆盖检查后，再结合 LoRA、低比特、稀疏注意力或跨分辨率方案。直接同时加入多项优化，会难以判断失真是训练、参数化还是推理实现导致的。'),
 h('第三轮验证真实产品约束'),
 p('增加长提示、不同宽高比、异常镜头、运行中提示更新和连续请求。记录冷启动、预热后性能、显存峰值和缓存状态。若用途要求稳定文字、精确计数或可靠物理过程，应允许更多步或设置质量回退，而非把最少 NFE 当成唯一目标。'),
 small('建议按可复现性和任务匹配度选择起点；不把最新论文或某个单项最高分自动视为最佳工程方案。'))

page('19 最近两个月的前沿清单',
 p('以下重点覆盖 2026 年 9 至 10 月，并补充值得联读的 2026 年路线。多数最新条目是预印本，适合跟踪与复现，证据成熟度弱于已重复验证的基础配方。'),
 t(['时间','工作','新增关注点'],[
 ['09-10','Uncertainty DMD [27]','首 chunk 不确定性与生成多样性'],
 ['09-24','SAP-DMD [13]','频谱振幅处理，改善更新方向的频率平衡'],
 ['09 月','DyMD [29]','视频动态的显式分布监督'],
 ['09-28','Elastic Forcing [30]','MMD 样本匹配，取消后训练在线双 score'],
 ['09-28','Projected DMD [31]','一项投影去除部分 critic 误差，扩展至音视频'],
 ['09-29','LongLive-Plug [32]','少步与长上下文蒸馏的适配器复用'],
 ['10-01','DMAD [14]','直接估计密度比，替代辅助 score 拟合'],
 ['10-07','Multi-Bandwidth DMD [36]','DMD 与 Drifting 的力场联系及多带宽学习'],
 ['10-08','Data-Space Iteration [37]','在数据空间迭代，而非依赖固定 flow 时间表']],[58,171,262]),
 h('理论联系应保留成立条件'),
 p('MBDMD 将教师速度或噪声场转换为吸引场，并从生成分布估计排斥场，讨论与 Drifting 的等价关系。这个关系依赖具体场构造和估计条件，不能理解成所有 Drifting 方法与所有 DMD 实现无条件相同。[36]'),
 h('最新不等于已经普适'),
 p('Data-Space Iteration 主要在 ImageNet256 等设置检验共享生成器的数据空间迭代；不能据此推断大型文生图或长视频已经得到同等验证。SAP-DMD 的部分对照去掉 GAN，也不能作为对完整 DMD2 的普遍结论。[37][13]'),
 h('相邻路线仍要保留'),
 p('Seaweed APT2 是值得比较的实时对抗后训练路线，其官方页报告 8B、736×416、单 H100、24 FPS 的流式 I2V。但它以 GAN 后训练为主，不能因为同样少步或实时就归入严格 DMD。[39]'))

page('附录 开源资产与可复现性',
 p('公开程度按本次查阅时已核验的内容描述。能下载权重、能执行推理、能重新训练出论文结果，是不同层级。正式使用前仍应固定仓库版本与模型权重。'),
 t(['项目','已核验入口或资产','使用时的边界'],[
 ['DMD2 [02]','官方训练实现与 SDXL 权重','适合做分布蒸馏基础基线'],
 ['SenseFlow [04]','作者项目页与模型入口','大规模 flow 的训练资源要求较高'],
 ['DP-DMD [10]','训练 评测脚本和 SD3.5-M 权重','验证首步梯度截断与多样性'],
 ['CDM [12]','官方代码','复现对照须保留 image-free 条件'],
 ['DMDR [07]','官方 ImageNet 训练演示','不等于完整大规模文生图配方'],
 ['CausVid / Self Forcing [15][16]','官方实现及相应模型入口','区分原始 CausVid 与 Wan 移植'],
 ['Causal Forcing 系列 [18][19]','训练推理代码及 1/2/4 步权重','首帧步数与硬件计时条件需一致'],
 ['LongLive / rCM [17][20]','官方训练与推理实现','包含缓存 并行与 JVP 等基础设施'],
 ['DFD / CoDMD [26][28]','训练 推理及 checkpoint 入口','明确参考数据与提示集'],
 ['SGMD [24]','LightX2V 中具体 trainer 实现','论文使用的提示数据并非全部公开'],
 ['PDMD / DMAD [31][14]','作者代码与模型入口','最新结果仍需独立复现'],
 ['FastVideo [35]','训练 配置 权重与部署文档','不同硬件需要对应精度和内核'],
 ['FSF-DMD / Uncertainty [25][27]','本次确认论文与相关项目材料','未确认完整官方训练代码可用性']],[125,194,172]),
 small('正文引用可点击跳转到文末；文末论文与代码地址均为外部可点击链接。数据和指标说明以原论文的具体版本为准。'))

for start,end in [(1,8),(9,16),(17,24),(25,32),(33,39),(40,43)]:
 page(f'参考资料 {start:02} 至 {end:02}',{'type':'refs','keys':[f'{i:02}' for i in range(start,end+1)]})

path=Path(__file__).with_name('dmd_report_content.json')
path.write_text(json.dumps({'pages':pages,'refs':refs},ensure_ascii=False,indent=2),encoding='utf-8')
print(f'{len(pages)} planned pages, {len(refs)} references; {path}')
