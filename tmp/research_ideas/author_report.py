from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
pages=[]
def p(s): return {'type':'p','text':s}
def h(s): return {'type':'h2','text':s}
def small(s): return {'type':'small','text':s}
def call(s): return {'type':'callout','text':s}
def f(s): return {'type':'formula','text':s}
def tb(headers, rows, widths=None):
    d={'type':'table','headers':headers,'rows':rows}
    if widths: d['widths']=widths
    return d
def page(title,*blocks,kicker='',cover=False):
    pages.append({'title':title,'kicker':kicker,'cover':cover,'blocks':list(blocks)})

page('图像 / 视频推理加速\n研究 Idea 与实验路线',
    p('基于既有 DMD 阅读方向的独立选题探索\n资源约束：8 × RTX 4090 / 4 × A800 80GB\n调研截止：2026 年 10 月 15 日'),
    call('建议先做「预测式 critic 跟踪」的机制验证，以 2 步生成质量为最终目标；并行做「面向后续影响的步数分配」离线诊断。固定历史的分叉覆盖蒸馏列为条件性备选，暂不直接立项。'),
    tb(['候选','核心研究问题','当前判断'],[
        ['A 预测式 critic 跟踪','能否预测学生更新引起的分布变化，减少少步训练中的错误监督？','优先证伪；机制较清晰；技术风险较高'],
        ['B 后续影响驱动的步数分配','哪一段多算两步，能真正改善后面的视频？','直接提速路线；先验证 oracle 空间'],
        ['C 固定历史的分叉覆盖','少步学生是否丢掉同一历史下仍合理的多种未来？','有价值的迁移假设；已有强近邻']
    ],[111,241,139]),
    p('本报告交付可证伪的研究假设、最近邻排重、算法细节、资源账本和实验顺序。推荐强度代表目前证据下的研究判断，不代表已经验证的论文结果。'),
    small('证据分为：公开论文与源码事实；本报告推导；已完成的小型数值检查；尚未执行的大模型实验计划。本次未连接你的 GPU 集群，也未声称完成图像或视频模型训练。'),
    cover=True)

page('01 选题判断与阅读导航',
    h('与你之前的方向保持一致，但把问题再收窄一层'),
    p('已查看工作区的 DMD 综述内容及 10 月 14 日组会材料。你主要关注 DMD / DMD2、SenseFlow、rCM / sCM、AC-DMD / RTDMD、SGMD、CoDMD、DyMD 和 Flash-DMD；共同主线是少步蒸馏中的监督质量、fake score 跟踪、运动和多样性。'),
    p('本报告以生成式图像和视频为范围，优先视频、辅以图像机制验证。排除硬件设计、CUDA 内核、量化工程和通信系统作为论文贡献；显存与训练实现只用于判断能否落地。'),
    tb(['阅读位置','内容'],[
        ['02-03','排重结果：哪些直觉已经有人做，哪些新论文改变选题判断'],
        ['04-07','主线 A：预测式 critic 跟踪，数学机制、实现和退出条件'],
        ['08-09','路线 B：基于后续影响的自适应去噪步数'],
        ['10-12','备选 C：条件分叉覆盖及其明确的新颖性限制'],
        ['13-16','资源、baseline、评价、阶段计划'],
        ['17 及文末','立项建议、证据边界和可点击参考来源']
    ],[93,398]),
    h('三条立项标准'),
    p('第一，核心机制必须与最近邻有可观察的区别；第二，能在投入大规模训练前设计否定假设的实验；第三，最终要改善质量与延迟的关系。只让 4 步模型分数升高，应称少步质量改进；只有相同质量下减少步数或实际耗时，才完成推理加速论证。'),
    small('检索采用论文原文、作者项目和官方仓库，重点核验与候选机制最接近的工作。检索未发现完全相同方法不等于保证全新；开题和投稿前仍需重新做引用链检索。'))

page('02 排除看似新颖、实际已有的方案',
    tb(['最初可能想到的做法','已覆盖的最近邻','本报告的处理'],[
        ['按采样步数给 fake score 加条件','TDM 已将目标采样步数 K 同时输入学生与 fake score。[06]','淘汰为独立创新'],
        ['过滤 critic 残差方向','PDMD 已做更新方向的残差投影。[05]','作为 A 的基线'],
        ['给视频加跨帧 / 跨样本关系损失','CoDMD 已做关系结构匹配。[10]','不重复包装'],
        ['首 chunk 加随机扰动保多样性','Uncertainty DMD 已处理首步条件和 cache 随机性。[09]','作为 C 的基线'],
        ['对初始噪声做有限差分响应匹配','GAD 已对齐教师 / 学生的噪声 JVP。[18]','不能单靠换成视频立项'],
        ['少步蒸馏结合可学习缓存','DisCa 已研究蒸馏兼容缓存；AutoTarget 分析 solver-facing 缓存目标。[20][21]','不作为模型创新主线'],
        ['把未来损失传回早期 chunk','Connected Self Forcing 已用 shortcut gradient replay。[22]','不主张首次跨 chunk 反传'],
        ['少步模型直接复用原有 LoRA','DART 已研究坐标迁移与目标日程校准。[30]','不作为本轮主选']
    ],[161,218,112]),
    h('这些淘汰结果意味着什么'),
    p('高质量选题需要指出一个现有目标或训练闭环仍无法解释的现象，再给出针对性机制。单纯叠加损失、缓存、路由或 LoRA，实验即使有效，也很难证明独立贡献。'),
    small('表中是对“能否单独作为新贡献”的判断，不是否定已有方法的价值。引用对应的具体机制已查原文或作者实现；本表没有把不同论文的分数拼成横向排名。'))

page('03 当前前沿：真正要正面对比什么',
    tb(['工作','本次核查到的机制','对选题的约束'],[
        ['SGMD [04]','联合改变学生 / fake 更新，减少高频 fake 更新依赖。','A 不能只声称减少 critic 次数'],
        ['AnyFlow [07]','flow map 与 on-policy 训练支持多个推理步数。','B 不把任意步数能力当创新'],
        ['CMD / Salt++ [08][32]','使生成、教师评分和 fake 训练的历史上下文更一致。','C 不主张首次按真实学生 prefix 蒸馏'],
        ['Elastic Forcing / ViRDM [12][13]','样本或表示空间分布匹配，减少在线 teacher / critic 依赖。','C 必须比较组内匹配与全局匹配'],
        ['DistillAlign [14]','协调 consistency 与 DMD 的覆盖 / 寻模目标。','仅“兼顾质量与多样性”不足'],
        ['Surprise Forcing [15]','首轮输出的时间变化启发式，选择两步或四步。','B 必须超过其局部难度信号'],
        ['ForcingBooster [16]','自适应 patch 去噪，是 10 月的新近邻。','B 不能把空间非均匀预算当新概念'],
        ['SMD [23]','10 月 13 日预印本：on-policy velocity 回归及训练期 proxy。','A 还要问保留 critic 是否划算']
    ],[126,214,151]),
    p('额外发现：气象集合预测蒸馏已经对同一历史下的教师 / 学生多样本使用 energy distance。[31] 因而 C 的损失形式本身已有先例，原创空间只能来自视频特有的诊断、训练分布和质量-步数收益。'),
    small('本报告中的新论文按预印本处理；未以网页自报的接收状态作为论证依据。重点是核心机制与实现可得性，而非论文名或总分。'))

page('04 主线 A：预测式 critic 跟踪',
    call('问题：fake score 总在拟合已经改变的学生分布。与其等它重新追上，能否根据学生刚发生的更新，先预测下一分布的 score，再用少量去噪训练校正？'),
    h('为什么值得先验证'),
    p('DMD 的学生梯度依赖教师与 fake score 的差。若 fake 的误差来自对旧学生分布的滞后，它会在极少步、分布变化较快时污染监督。DMD2 增加 fake 更新预算，SGMD 改变耦合更新，PDMD 处理残差方向；A 则显式估计一次学生参数更新造成的样本输运。[01][04][05]'),
    h('核心假设与贡献边界'),
    p('假设：少步蒸馏中有一部分 critic 误差是可由相邻学生版本的样本漂移预测的，而非不可预测的拟合噪声。若成立，训练期加入输运预测器，可在低 fake 更新预算下改善 2 步学生的覆盖、语义和运动质量。'),
    p('拟议贡献是训练迭代轴上的 score 预测及其误差诊断。这里的“时间”指学生从一次 optimizer update 到下一次的变化；它不同于扩散噪声时间 t，也不同于 AC-DMD 的跨扩散时间一致性。连续性方程与一阶展开本身是已有数学工具。'),
    tb(['维度','预期机制 / 必须证明的结果'],[
        ['推理路径','仍只有学生；预测器、旧 critic 和漂移估计均在训练时使用'],
        ['加速主张','证明 2 步达到原 4 步的可接受质量，或改善相同延迟下的质量前沿'],
        ['不能替代的证据','critic MSE 下降不自动等于生成更好；训练更快不自动等于推理更快'],
        ['当前评级','机制区分较清晰，值得优先证伪；新颖性与大模型收益待验证']
    ],[100,391]))

page('05 A 的数学机制：预测分布如何移动',
    p('固定文本条件 c 和噪声时间 t。令训练迭代连续坐标为 u，x_t(u) = alpha_t G_theta(u)(z,c) + sigma_t epsilon。相邻学生使用相同 z 和 epsilon，测到的差值主要来自参数变化。'),
    f('v_u,t(x,c) = E[ d x_t(u) / du | x_t(u)=x, c ]\npartial_u q_u,t(x|c) = - div_x( q_u,t(x|c) v_u,t(x,c) )'),
    p('在密度光滑、边界通量适当、条件漂移和旧 score 精确的理想情形下，记 s_u,t = grad_x log q_u,t，可推出：'),
    f('partial_u s_u,t = - grad_x [ div_x v_u,t + v_u,t^T s_u,t ]\ns_pred = s_old - Delta_u * grad_x[ div_x v + v^T s_old ]'),
    p('在固定坐标 x 上预测新 score。Delta_u 可吸收到“每次更新的位移场”中，避免实现中重复乘步长。冻结旧 critic 后对输入求导；预测目标停止参数梯度，再用新学生样本的 DSM 去噪回归校正。'),
    h('第一版只做局部仿射漂移'),
    f('v(x) = Bx + b,  B = U A U^T\ns_pred = s_old - Delta_u * [ B^T s_old + J_s_old(x)^T v(x) ]'),
    p('仿射场的 div(v) 是常数，去掉其梯度，可用低秩 B 与一次输入 VJP 计算修正。式中是 J_s 的转置；理想 score 的 Jacobian 对称，但实际网络不应擅自假定对称。低秩 / 分块参数化仅是可行性近似，表达不足时必须在诊断中暴露。'),
    small('成立限制：只是一阶、小更新近似；旧 critic 误差、条件漂移估计误差和曲率余项都会留下。t 接近零时需更谨慎，不能称无偏估计，也不能由此推导训练必然收敛。'))

page('06 A 的可实施版本与代价',
    tb(['步骤','操作','关键控制'],[
        ['1 测量漂移','更新学生前保存少量 probe 输出；更新后用同条件、同噪声重算。','避免第二份完整旧学生；记录真实额外 rollout 成本'],
        ['2 拟合条件位移','以旧query x_old为输入，回归(x_new-x_old)/Delta_u。','需对c条件化；在新query处同时求v与旧score，使用固定坐标公式'],
        ['3 构造 score 预测','旧 critic 冻结参数但保留输入梯度，算输运修正。','保留旧 critic 快照或先缓存预测目标；禁止目标随优化漂移'],
        ['4 校正 fake','用预测目标初始化式监督，并加入新分布 DSM。','先测试弱权重；DSM 保持为锚点，避免累积预测误差'],
        ['5 更新学生','下一次学生更新使用校正后的 fake。','冻结 fake 参数；沿用原 DMD 梯度接口'],
        ['6 可信区间回退','大漂移 / 高 held-out 误差时禁用修正，退回普通 DSM。','回退本身作为消融，不能隐藏失败样本']
    ],[70,239,182]),
    p('先收集已有 DMD 训练的相邻 checkpoint，离线冻结每对做漂移审计，再做在线实验。首版固定少量prompt /类别并重复多个z；共享B_t与条件b_t(c)仅是受限近似，必须测held-out误差。图像机制通过后才迁移Wan1.3B。'),
    p('必须计入：probe 重新采样、漂移拟合、输入 VJP、旧 critic 权重或预测标签缓存。没有训练期“零开销”的承诺。若新增成本比多做一次 DSM 更大，却没有更好的 2 步质量，应停止。'),
    small('实现候选可以把预测目标加入 L_fake = L_DSM + lambda * ||s_phi(x_new)-stopgrad(s_pred(x_new))||^2。该辅助项有偏，只在预测经验证可靠时使用；它不是将网络参数按 score 公式直接平移。'))

page('07 A 的最小证伪实验',
    h('先确认误差来源，再投入模型训练'),
    tb(['阶段','实验设计','退出条件'],[
        ['A0 解析检查','高斯平移 / 缩放验证一阶 score 输运；混合分布再验证局部失效。','连精确旧 score 下的阶数都不对，先修推导'],
        ['A1 冻结漂移审计','小图像学生的相邻 checkpoint；对新分布额外训练高预算 reference critic，只作离线近似参照。','滞后误差太小，或漂移预测无法优于直接复用旧 score'],
        ['A2 受控在线实验','2 步固定学生；DMD2 r=1 / r=5、SGMD、PDMD、预测+DSM对照。','等 GPU 小时没有覆盖 / 语义收益，或只在训练集出现'],
        ['A3 视频确认','固定教师与1.3B学生，比较2步 / 4步，保留相同初始化与评测。','运动增强来自伪影，或2步不能逼近可接受4步质量']
    ],[72,256,163]),
    h('哪些消融真正能解释贡献'),
    p('比较stale critic、权重EMA / 简单外推、输运无DSM、输运+DSM、同成本额外DSM。另分别输运高预算旧reference与实际旧critic：若前者有效、后者无效，瓶颈是旧critic误差或Jacobian噪声，不应盲目扩展漂移模型。reference不是真值，须以两种容量 /预算复核。'),
    p('主结果至少包含“相同学生更新数”与“相同 GPU 小时”两种预算。若 r=5 明显胜出，说明预测器尚不能替代正常拟合；若只有训练速度提高，按训练效率论文定位，不把结果写成额外推理加速。'),
    call('A 的第一道门槛：预测所带来的误差下降，必须超过同成本普通 DSM 的下降。先证实这件事，再训练视频。'),
    small('建议门槛是实验决策规则：跨 3 个训练 seed 的主要指标方向一致，按 prompt 聚类 bootstrap 的置信区间支持收益；不预设必须获得某个漂亮的加速倍数。'))

page('08 路线 B：多算两步，要改善后面的画面',
    call('研究问题：当前 chunk 看起来难，不等于多算它最有价值。能否学习“多两步对接下来几段视频的实际改善”，把预算放在影响未来状态的关键片段？'),
    p('最近邻 Surprise Forcing 用当前 chunk 首次去噪后的邻帧距离判断是否跳步；PARE 选择网络模块，ForcingBooster 做 patch 层面的预算调整。B 的潜在差异是监督标签：通过受控分叉，估计一次计算选择对后续内容的增量影响，而非仅预测局部误差。[15][16][17]'),
    h('反事实收益标签'),
    p('在同一历史 H 和相同首个去噪状态下，分叉执行短日程与长日程。它们完成后，分别重建各自 cache；随后都用固定日程续跑 L 个 chunk，并使用对应的共同随机数。'),
    f('Delta_Q_L(H,z) = Q( Y_(current:current+L)^(long) )\n                    - Q( Y_(current:current+L)^(short) )\nchoose long if gain_head(features, remaining_budget) > lambda * Delta_cost'),
    p('Q 应组合指令遵循、主体保持、动作合理性和伪影约束，权重只在验证集确定。高步结果不自动是正确答案，不能只以到高步输出的 LPIPS 当质量标签。L=0 与 L>0 的区别是本路线最重要的消融。'),
    h('推理只保留一个轻量预测头'),
    p('预测头读取已完成的公共首个去噪调用的 hidden feature、历史摘要和剩余预算。无需在测试时展开两个分支，也无需额外教师。整个新方法仍属于模型与采样算法；不依赖新内核或特定硬件机制。'),
    small('必须首先确保两条日程共享同一初始调用。对 flow map，首跳的目标时间 r 可能不同；仅“同一噪声”不保证首调用相同。不能把不兼容日程硬拼再声称免费共用特征。'))

page('09 B 的实验：先找分配价值，再训练路由',
    h('最小版本与对照'),
    p('第一版采用共享首调用的2 / 4次调用日程，锁定checkpoint和cache策略。AnyFlow需先选择共享首跳的合法时间网格并验证质量；Surprise的{0,3} / {0,1,2,3}日程则原样保留转换和重新加噪规则。[07][15]'),
    tb(['比较','回答的问题'],[
        ['固定2步、固定4步、随机混合','任何收益是否仅来自平均步数改变？'],
        ['Surprise 难度分配','是否优于已有几乎免费的局部启发式？'],
        ['L=0 的局部收益头','收益标签本身有效，还是必须看后续？'],
        ['L>0 的后续收益 oracle','已知分叉结果时，固定总预算到底存在多少分配空间？'],
        ['同特征 / 同参数量的学习头','收益是否仅来自增加了一个可训练模型？'],
        ['真实线上策略完整 rollout','离线 oracle 优势能否迁移到策略自己访问的历史？']
    ],[180,311]),
    p('分叉收益是固定后续策略下的局部效应，不能把参考轨迹所有chunk的收益简单排序当全局最优：早期选择会改变后来历史。先做固定历史诊断，再在真实访问的H重新分叉，运行完整lookahead oracle；它是诊断参照，不是可部署速度。'),
    p('随机数按chunk、噪声时刻、seed索引。总预算包含公共首调用，并为剩余chunk预留最低2步成本。先64个开发prompt，再200个独立验证prompt；视频先5-10秒，再30-60秒。补UnStep低步数基线并锁定其cache修复开关，不引用含runtime优化的总加速。[33]'),
    call('退出条件：固定总 NFE 下，后续收益 oracle 也不能显著优于随机 / Surprise，立即停。若 oracle 有空间而学习头无收益，再查标签噪声、历史分布偏移和特征不可辨识，不急着扩大模型。'),
    small('算账示例：半数chunk用2步、半数4步，平均3步；仅采样调用的理想比为4/3。若每chunk还需一次完整clean-cache调用，DiT比值变为5/4；端到端收益还会缩小。这不是预计实测成绩。'))

page('10 备选 C：固定历史后，还剩几种合理未来',
    p('同一 prompt 换 seed，可能只改变了第一段场景。真正需要检验的是：把已经生成的历史 H 完全固定后，重新采样后续噪声，学生是否仍能产生教师允许的多种合理动作。这个问题比总体 diversity 更具体。'),
    h('为什么整体分布看起来正常仍可能有问题'),
    f('P(H=-1) = P(H=+1) = 1/2\np_T(Y | H=-1) = p_T(Y | H=+1) = 0.5 delta_-1 + 0.5 delta_+1\nq_S(Y | H=-1) = delta_-1;   q_S(Y | H=+1) = delta_+1\np_T(Y) = q_S(Y), but q_S(Y|H) != p_T(Y|H)'),
    p('这个可精确计算的示例中，整体 energy distance 为 0，同一历史内的平均 energy distance 为 1。它证明“混在一起评估会掩盖条件坍缩”在逻辑上可能；并不证明真实视频模型一定有同样问题。'),
    h('新颖性限制必须先讲清楚'),
    p('气象集合预测蒸馏已经采用同一历史下的多教师 / 多学生样本 energy distance；CMD 和 Salt++ 已做历史条件对齐。因此，“条件分布匹配”与下面的 U-statistic 都不能作为原创算法声明。[08][31][32]'),
    p('C 只在证实视频中特定的缺口后值得做：on-policy 学生历史下，少步蒸馏损失了教师原本具有的动作分叉能力，而当前噪声扰动或上下文对齐无法恢复它。潜在贡献是诊断、训练分组方式及相同质量下的更少步数。'),
    call('当前结论：保留为有条件备选。若教师在固定 prefix 下本来几乎确定，或者现有 Uncertainty DMD / CMD 已消除缺口，就不应人为制造“更随机”的视频。'))

page('11 C 的具体损失与实现',
    p('从学生rollout取H并stop-gradient；教师用自己的模型重新编码同一H。teacher target及表征中的H均固定。分别生成K=4个教师后续与K=4个学生后续，噪声组独立。令u_i=phi(H,Y_i^S)、v_j=phi(H,Y_j^T)，仅同H组内计算。'),
    f('ED_H = 2/K^2 * sum_(i,j) d(u_i,v_j)\n       - 1/[K(K-1)] * sum_(i!=j) d(u_i,u_j)\n       - 1/[K(K-1)] * sum_(i!=j) d(v_i,v_j)\nL_student = L_base + lambda * E_H[ ED_H ]'),
    p('欧氏范数的 energy distance 是可用的起点；有限样本无偏估计可能为负，不能随意 clamp 后仍称无偏。teacher-only 项对学生梯度为零，但诊断时保留。固定表征上的分布一致也不保证像素或动作全分布一致。'),
    h('表征先简单、可微，评估再独立'),
    p('第一版用冻结的视频 latent / 表征网络的时间差分特征，并同时保留静态主体特征约束。训练时需要对学生 latent 可微；光流 / 轨迹追踪先只作独立评价，避免不可微评估器让训练损失没有梯度。检查镜头运动、闪烁和主体漂移是否在作弊。'),
    p('不能把所有样本都减去同一个 prefix 预测值就声称得到新目标：同组欧氏距离对公共平移不变。若做相机补偿或非线性运动表征，需要真正改变特征并单列消融。'),
    h('低显存实现'),
    p('顺序生成各分支并保存特征，再replay反传。可独立随机挑一个学生分支，用K倍该分支完整特征梯度估计固定H目标的梯度，勿遗漏对称pair项。这仅对固定H及选定的截断图成立，不是整个on-policy历史分布的无偏梯度；减少显存但不减少采样总成本。'),
    small('可核验入口：Causal-Forcing 的 inference(noise, text_prompts, initial_latent=H, return_video=False) 支持视频续写。[27] 初版只生成短 suffix；教师步数可先16/32，再用50步做收敛敏感性检查，不能把尚未收敛教师当绝对真值。'))

page('12 C 的证据门槛与样本预算',
    tb(['诊断 / 消融','必须观察什么'],[
        ['固定 H，多未来 seed','教师有合理的条件分叉，学生明显更集中；同时检查语义 / 伪影'],
        ['全局匹配 vs 同 H 组内匹配','同样teacher样本数和训练时间，条件分组本身有增益'],
        ['真实历史 vs 学生历史','on-policy H 的收益不能只靠更多样本解释'],
        ['CMD / Uncertainty / DistillAlign','缺口在近邻方法上仍存在，拟议方法才能有独立价值'],
        ['K=2 / 4 / 8','收益不是不断加teacher采样成本的结果'],
        ['2步 / 4步','证明覆盖的恢复允许降低步数；否则只报告质量改善']
    ],[182,309]),
    p('第一轮计划：64 个分层 prompt × 2 个 prefix × 4 个未来 seed，分别测教师与学生，共 1,024 条短 suffix；先用其中 8 个 prompt 测试续写正确性和教师分叉是否合理，再决定是否跑全量。样本数量是资源计划，不是统计功效保证。'),
    p('条件覆盖可以用同 H 教师样本的最近邻召回作代理，但同时报告学生到教师的距离、主体一致性和人工动作合理性。教师只采4个样本时，这只是有限集合覆盖，不能称捕获了完整真实分布。'),
    h('最低可信实验结构'),
    p('训练 / 验证 / 测试按 prompt 和来源场景分离；同视频切出的 prefix 不得跨集合。保留至少三类历史：自然静态场景、合理多未来动作、教师难以续写的异常历史。第三类单独报告，避免把教师对坏 prefix 的随机错误学成“多样性”。'),
    call('只有当“视频特有条件缺口”与“条件分组带来的收益”同时被证实，C 才升级立项。否则，它只是把已有气象蒸馏损失迁移到视频，不足以支撑强原创主张。'))

page('13 算力账本：先按模型状态估下限',
    p('推荐 4 × A800 80GB 作为 1.3B 学生训练主力，8 × 4090 用于并行采样、评估和小图像模型机制实验。两组卡分别规划，不把它们当作同一台 12 卡训练机；也不假设两组资源必然同时可用。'),
    f('M_state ~= 16 * (P_student + P_fake) + (2 to 4) * P_teacher\n            + M_EMA  [bytes; excludes activations and temporary buffers]'),
    tb(['配置','简化全局状态账本','判断'],[
        ['1.3B G + 1.3B F + 1.3B T','44.2 GB起','适合受控机制实验；教师质量需验证'],
        ['1.3B G + 1.3B F + 14B T','69.6-97.6 GB','更接近Self-Forcing公开配方；仍须实测峰值'],
        ['8B G + 8B F + 8B T','272 GB起','4卡理想每卡已68GB，未含激活；不作起步'],
        ['12B G + 12B F + 12B T','408 GB起','超4×80GB总量；不作常规全参起步']
    ],[188,125,178]),
    p('上述按常规 Adam 的近似 16 bytes / 可训练参数估算，冻结教师按 2-4 bytes。使用十进制 GB，工具的 GiB 需换算。FSDP 只分摊部分常驻状态；block all-gather、梯度、KV、VAE、文本编码器、EMA 和 checkpoint 保存都会抬高单卡峰值。'),
    h('不要被推理显存或启动命令误导'),
    p('Wan1.3B的官方推理显存不能用于估算DMD训练。[24] Self-Forcing当前配置使用14B教师。[25] DMD2官方提示其特定LoRA实现仍有速度和显存问题，不能把LoRA当作必然省资源的保证。[01]'),
    small('硬件在这里仅决定实验规模，不是研究贡献。Linux / CUDA、系统内存、磁盘与多卡互连需在实际机器确认；未获集群访问，因此不提供伪精确的“几小时训练完成”承诺。'))

page('14 Baseline 与训练移植的具体顺序',
    tb(['研究线','最小模型 / 入口','进入大实验前必须确认'],[
        ['A 图像机制','DMD2 的小图像配置或SD1.5；固定2步与教师。','旧新学生漂移可测；critic输入梯度可用；同成本DSM基线'],
        ['A 视频迁移','Wan1.3B学生 + 固定1.3B或14B教师。','所有方法使用同一teacher；不能跨教师比较方法效果'],
        ['B 分配诊断','支持短 / 长合法日程的同一因果checkpoint。','公共首调用、cache写入、重新加噪与位置编码一致'],
        ['C 条件续写','Causal-Forcing公开续写接口；固定教师与学生。','教师用自己的编码器 / KV；initial_latent确实锁住prefix']
    ],[78,206,207]),
    h('先复现公开 checkpoint，再复现本地配方'),
    p('现成权重能推理，不等于训练移植已经成功。先跑固定 prompt / seed 的采样回归，再以小 batch 重现方向正确的曲线。官方配方和小资源改版要分开命名；若换了教师、分辨率或训练数据，不能声称复现了论文表中的绝对分数。'),
    h('Self-Forcing 有两个容易踩的计数问题'),
    p('本次检查的 trainer 每轮更新 fake，而学生按比例更新；600 个循环不能直接当 600 次学生 optimizer step。配置里的 total_batch_size 也不会自动替你完成梯度累积。移植时固定真实学生 / fake 更新次数，并检查 EMA、学习率日程和 checkpoint 恢复。[25][26]'),
    f('effective_batch = GPU_count * micro_batch * accumulation\nT_total ~= U_G*t_G + U_F*t_F + T_eval + T_save + T_precompute'),
    p('预热后测20-50个稳定完整更新，分别计学生、fake、教师CFG和保存峰值。t_G / t_F 若已包含梯度累积，不再重复乘 accumulation。离线标签、教师参考视频与诊断 critic 的成本均单独列账。'),
    small('数据可优先沿用baseline已有prompt /初始化资产，但需核对可用性与许可；只含prompt的DMD阶段与需要图像 /视频数据的初始化阶段不能混称data-free。'))

page('15 评价：避免把静止、闪烁或低质量当进步',
    tb(['需要回答的问题','指标与协议'],[
        ['画面与文字仍对吗？','VBench质量 / 语义分项；图像用GenEval关系、计数、属性分项。[28][29]'],
        ['运动更合理，还是只更大？','dynamic degree与motion smoothness、flickering、human action联合看；加盲评动作合理性'],
        ['多样性来自哪里？','同prompt多seed、同prefix多未来seed分别报告；背景差异与主体动作差异分开'],
        ['是否真的更快？','同设备同软件，denoiser / VAE / 完整链路P50和P95；流式首chunk延迟与稳态吞吐'],
        ['训练是否更划算？','相同学生更新数、相同teacher调用、相同GPU小时；三者不能混为一个预算'],
        ['结果是否稳定？','至少3个训练seed并报波动；以prompt为外层cluster做bootstrap，同prompt下prefix / seed不算独立样本']
    ],[131,360]),
    h('对齐所有影响速度与质量的变量'),
    p('固定模型、帧数、分辨率、CFG、负提示词、seed、VAE、精度、attention 实现和 cache 策略。CFG 的两次模型调用要算入 NFE。采样步数、完整 transformer 调用数和部分执行成本分别标注。禁止用高分辨率教师耗时对比低分辨率学生来制造加速。'),
    p('GPU计时先预热、显式同步，探索期至少30次；正式P95建议至少100次并给区间，冷启动另列。训练期多分支和诊断费用单列。FID / FVD样本数和参照分布须一致，教师参照与真实数据参照分开命名。'),
    h('预先登记可接受质量区间'),
    p('用验证集与重复测量确定质量容忍度 delta_Q、语义容忍度 delta_S 以及最低运动合理性要求，再比较质量约束下最快配置。不要在测试后挑最好看的指标定义“无损”。探索期小样本用于筛查，正式结论需要完整官方评测和人工核验。'))

page('16 六周路线：按证据放大投入',
    p('以下是工作排程建议，假定能使用现有预训练权重；不是训练收敛或论文成功的承诺。每个阶段都允许提前结束。先让 A 与 B 的低成本诊断竞争，再只放大胜出的一个。'),
    tb(['阶段','工作与产物','放大 / 停止标准'],[
        ['第1周','环境pilot；复现推理；A解析与漂移审计；B少量分叉标签。','确认显存、更新计数、日程兼容性；测到研究对象确实存在'],
        ['第2周','A小图像在线对照；B固定预算oracle；C仅8-64prompt诊断。','A胜过等成本DSM或B有稳定oracle空间；C有明确条件覆盖缺口'],
        ['第3-4周','选择一条主线迁移 / 微调1.3B视频；完成核心消融。','同教师同初始化下，质量-延迟前沿改善'],
        ['第5周','补3个训练seed、独立测试集、长视频或图像迁移。','收益不依赖单seed或一个benchmark'],
        ['第6周','速度实测、盲评、失败分析、复现脚本和论文图。','主张与证据一致；决定继续扩大或转向']
    ],[68,248,175]),
    h('用配额控制预算，而不是先猜总耗时'),
    p('建议前10%的可用GPU小时用于baseline与profile，接下来20%用于A/B诊断；只有通过门槛的主线才获得余下70%。若A/B都未通过，不把剩余预算用于增加训练轮数，而是根据失败证据重选问题。'),
    p('4张A800优先给需要反向传播的训练；4090用于固定权重推理、多个seed与离线标签。没有必要为了“用满卡”把独立短任务改成难调试的异构联合训练。'),
    call('第一周最有价值的产物不是好看的生成样例，而是两张图：A的“新分布score误差-总训练成本”曲线，B的“固定预算oracle-局部启发式”前沿。它们能决定后续是否值得投入。'))

page('17 最终立项建议与证据边界',
    h('优先级一：A，作为原创算法主线的候选'),
    p('它与已有fake跟踪工作关联紧密，却提出更具体的可检验机制：用相邻学生的样本耦合，预测训练迭代轴上的score变化。先完成小模型的误差 / 成本验证；只有确实改善2步学生，才进入视频。当前最大风险是漂移场难拟合、旧critic误差被放大、VJP成本吃掉收益。'),
    h('优先级二：B，作为直接推理提速路线'),
    p('它可以先用冻结模型验证，不需要马上训练整个DiT。关键区别是后续影响的收益标签，而非简单motion-aware router。若oracle都没有优势，问题本身没有足够空间；若oracle很强而预测头弱，则值得研究可预测性与策略分布偏移。'),
    h('优先级三：C，先诊断再决定是否做'),
    p('固定历史的未来覆盖有实际意义，但条件energy蒸馏已有跨领域先例，视频上下文对齐也已有强工作。必须拿到视频特有的实证和严格消融，才能从“合理迁移”变成值得发表的贡献。它不是本轮最强的新算法承诺。'),
    tb(['本次已经完成','尚未完成'],[
        ['既有材料对齐、主要近邻原文排重、公开训练源码资源审计','全量引用链查新与所有baseline的运行复现'],
        ['高斯score输运与条件分布混合盲区的小型数值 / 精确检查','真实图像 / 视频模型训练、GPU峰值与延迟测量'],
        ['可执行实验顺序、公式、指标与退出条件','任何“显著领先SOTA”或“保证几倍加速”的验证']
    ],[245,246]),
    small('研究判断：优先选择能用小实验迅速推翻的机制，而不是必须先耗完整预算才能解释成败的复杂系统。本报告不把未检索到同名论文当作新颖性证明。'))

refs={
 '01':{'title':'Improved Distribution Matching Distillation for Fast Image Synthesis (DMD2)','url':'https://arxiv.org/abs/2405.14867','meta':'基础方法；训练实现与Known Issues另见官方仓库。','code':'https://github.com/tianweiy/DMD2'},
 '02':{'title':'Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion','url':'https://arxiv.org/abs/2506.08009','meta':'学生历史rollout与少步因果视频baseline。','code':'https://github.com/guandeh17/Self-Forcing'},
 '03':{'title':'SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation','url':'https://arxiv.org/abs/2506.00523','meta':'与用户既有flow蒸馏方向关联；不作本报告新贡献。','code':'https://github.com/XingtongGe/SenseFlow'},
 '04':{'title':'SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion Distillation','url':'https://arxiv.org/abs/2605.30116','meta':'fake跟踪与耦合更新的必要近邻。','code':'https://github.com/ModelTC/LightX2V'},
 '05':{'title':'PDMD: Projected Distribution Matching Distillation for Video Diffusion Models','url':'https://arxiv.org/abs/2609.35768','meta':'2026-09预印本；残差方向投影。'},
 '06':{'title':'Learning Few-Step Diffusion Models by Trajectory Distribution Matching (TDM)','url':'https://arxiv.org/abs/2503.06674','meta':'重点核查§3.3：目标采样步数K输入student与fake。'},
 '07':{'title':'AnyFlow: Any-Step Video Diffusion Model with On-Policy Flow Map Distillation','url':'https://arxiv.org/abs/2605.13724','meta':'任意步数与因果模型入口；需核实具体checkpoint日程。','code':'https://github.com/NVlabs/AnyFlow'},
 '08':{'title':'Context-Matched Distillation: Teacher Causality for Autoregressive Video Distillation','url':'https://arxiv.org/abs/2608.13391','meta':'2026-08预印本；相同学生历史下的教师评分。'},
 '09':{'title':'Uncertainty DMD: Restoring Diversity in Few-Step Autoregressive Video Distillation','url':'https://arxiv.org/abs/2609.11265','meta':'2026-09预印本；初始条件与cache随机性。'},
 '10':{'title':'CoDMD: Copula-aware Distribution Matching Distillation for Fast Video Generation','url':'https://arxiv.org/abs/2606.21982','meta':'跨样本 / 跨帧关系约束；非本报告新提出。'},
 '11':{'title':'Salt: Self-Consistent Distribution Matching with Cache-Aware Training for Fast Video Generation','url':'https://arxiv.org/abs/2604.03118','meta':'自一致性与cache质量训练近邻。','code':'https://github.com/XingtongGe/Salt'},
 '12':{'title':'From Scores to Samples: Elastic Forcing for Autoregressive Video Generation','url':'https://arxiv.org/abs/2609.35491','meta':'2026-09预印本；样本匹配，注意marginal与conditional区别。'},
 '13':{'title':'ViRDM: Taming Representation Distribution Matching for Few-Step Causal Video Generation','url':'https://arxiv.org/abs/2609.28923','meta':'2026-09预印本；表示分布匹配与低内存训练路径。'},
 '14':{'title':'DistillAlign: Coordinating Mode Covering and Mode Seeking in Autoregressive Video Distillation','url':'https://arxiv.org/abs/2607.26811','meta':'2026-07预印本；consistency与DMD覆盖 / 寻模协调。'},
 '15':{'title':'Surprise Forcing: What to Remember, When to Skip in Long Video Generation','url':'https://arxiv.org/abs/2607.18436','meta':'2026-07预印本；§3.4局部难度与2/4步日程。'},
 '16':{'title':'ForcingBooster: Adaptive Patch-wise Denoising for Fast Autoregressive Video Diffusion','url':'https://arxiv.org/abs/2610.12625','meta':'2026-10-08预印本；patch级自适应去噪近邻。'},
 '17':{'title':'PARE: Pruning and Adaptive Routing for Efficient Video Generation','url':'https://arxiv.org/abs/2605.27336','meta':'结构剪枝与输入 / 时间条件路由；B的算法范围比较。'},
 '18':{'title':'Restoring Initial Noise Sensitivity in Text-to-Image Distillation via Geometric Alignment','url':'https://arxiv.org/abs/2606.01651','meta':'GAD；已存在噪声方向响应 / JVP匹配。'},
 '19':{'title':'DyMD: Preserving Interaction Dynamics through Distribution Matching Distillation in Few-Step Video World Models','url':'https://arxiv.org/abs/2609.31349','meta':'2026-09预印本；重加噪与fake困难样本加权。'},
 '20':{'title':'DisCa: Accelerating Video Diffusion Transformers with Distillation-Compatible Learnable Feature Caching','url':'https://arxiv.org/abs/2602.05449','meta':'蒸馏兼容缓存的已存在路线。','code':'https://github.com/Tencent-Hunyuan/DisCa'},
 '21':{'title':'Rethinking What to Cache in Few-Step Diffusion Transformers: Solver-Aware Target Selection','url':'https://arxiv.org/abs/2610.03577','meta':'2026-10预印本；AutoTarget，缓存目标与solver等价性排重。'},
 '22':{'title':'Connected Self Forcing: Beyond Local Learning in Video Autoregression','url':'https://arxiv.org/abs/2610.12156','meta':'2026-10-08预印本；跨chunk梯度replay。'},
 '23':{'title':'Simplifying and Improving On-Policy Distillation for Few-Step Diffusion','url':'https://arxiv.org/abs/2610.18472','meta':'2026-10-13预印本；SMD，新近的score-free竞争路线。'},
 '24':{'title':'Wan2.1 official repository','url':'https://github.com/Wan-Video/Wan2.1','meta':'模型权重和推理资源说明；推理显存不等于训练显存。'},
 '25':{'title':'Self-Forcing official training configuration','url':'https://github.com/guandeh17/Self-Forcing/blob/main/configs/self_forcing_dmd.yaml','meta':'本次读取：real_name为14B，fake / generator为1.3B；main可能后续更新。'},
 '26':{'title':'Self-Forcing official trainer','url':'https://github.com/guandeh17/Self-Forcing/blob/main/trainer/distillation.py','meta':'本次读取：更新计数与梯度累积审计；main可能后续更新。'},
 '27':{'title':'Causal-Forcing official causal diffusion inference pipeline','url':'https://github.com/thu-ml/Causal-Forcing/blob/main/pipeline/causal_diffusion_inference.py','meta':'initial_latent视频续写接口；教师必须重建自己的KV。','code':'https://github.com/thu-ml/Causal-Forcing'},
 '28':{'title':'VBench official evaluation code','url':'https://github.com/Vchitect/VBench','meta':'视频质量、语义与运动分项；正式测试遵循同版协议。'},
 '29':{'title':'GenEval official evaluation code','url':'https://github.com/djghosh13/geneval','meta':'图像组合语义评价；作为A的图像机制验证补充。'},
 '30':{'title':'DART: Distillation-Aware Reparameterization for Training-Free LoRA Reuse in Few-Step Video Diffusion Models','url':'https://arxiv.org/abs/2609.20051','meta':'2026-09预印本；已有LoRA跨蒸馏日程复用研究。'},
 '31':{'title':'Diffusion Distillation for Efficient Weather Ensembles','url':'https://arxiv.org/abs/2608.27728','meta':'2026-08预印本；§2、§3.2条件多样本energy蒸馏，是C的实质近邻。'},
 '32':{'title':'Salt++: Context-Aligned Post-Training for Few-Step Streaming Multimodal Generation','url':'https://arxiv.org/abs/2609.36995','meta':'2026-09预印本；block-conditional KL与生成历史适配。'},
 '33':{'title':'UnStep: Training-Free Acceleration of Causal Video Diffusion with Fewer Steps Than Distillation','url':'https://arxiv.org/abs/2609.32518','meta':'2026-09预印本；减少步数并修复cache；总体速度还包括runtime优化。'}
}

checks=json.loads((ROOT/'toy_transport_results.json').read_text(encoding='utf-8'))
page('附录：本次完成的两个机制检查',
    p('以下均为CPU解析计算，不包含神经网络训练或视频模型推理。它们检验论证中的必要条件，不能证明候选方法的实际生成质量或加速收益。'),
    {'type':'image','path':str(ROOT/'toy_mechanism_checks.png'),'width':491,'height':218.2},
    h('A：一阶输运的局部阶数正确'),
    p('旧分布N(0,1)，新变量Y=(1+0.3*delta)X+0.7*delta。旧score与漂移场均精确。在新分布下计算RMSE，stale score约为O(delta)，预测后约为O(delta^2)。实测的是解析式的数值，不是训练收敛率。'),
    tb(['delta','旧score RMSE','预测score RMSE'],[[str(r['delta']),f"{r['stale_rmse']:.6f}",f"{r['predicted_rmse']:.6f}"] for r in checks['transport']['rows']],[90,200,201]),
    p('四个点的log-log拟合阶数分别为0.9951与1.9977。真实critic误差、拟合位移误差和非线性多峰变化尚未检验；不能把本表的误差倍数当模型加速倍数。'),
    h('C：边际相等不排除条件坍缩'),
    p('§10的两点分布，边际ED精确等于0，平均条件ED精确等于1。它只证明一种评测盲区可能存在；并不说明所有现有视频损失都忽略历史。'),
    small('复核资产：tmp/research_ideas/toy_transport.py 与 toy_transport_results.json。图中ED采用2E|X-Y|-E|X-X\'|-E|Y-Y\'|，不取平方根。'))

for n in range(4):
    keys=[f'{i:02d}' for i in range(n*8+1,34 if n==3 else n*8+9)]
    page(f'参考来源 {n+1} / 4',{'type':'refs','keys':keys},
         small('来源查阅日：2026-10-15。论文与作者源码属于外部证据；本报告提出的算法、优先级和门槛属于研究建议。代码main分支可变，实际实验开始时应记录commit、模型revision和数据版本。'))

(ROOT/'report_content.json').write_text(json.dumps({'pages':pages,'refs':refs},ensure_ascii=False,indent=2),encoding='utf-8')
print('Wrote',len(pages),'logical pages')
