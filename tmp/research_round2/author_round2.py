from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
pages=[]
def p(s):return {'type':'p','text':s}
def h(s):return {'type':'h2','text':s}
def small(s):return {'type':'small','text':s}
def call(s):return {'type':'callout','text':s}
def f(s):return {'type':'formula','text':s}
def tb(a,b,w=None):
    o={'type':'table','headers':a,'rows':b}
    if w:o['widths']=w
    return o
def img(name,height=220):return {'type':'image','path':str(ROOT/name),'width':491,'height':height}
def page(title,*b,cover=False):pages.append({'title':title,'blocks':list(b),'cover':cover})

page('图像 / 视频推理加速\n第二轮选题打磨',
    p('模型与蒸馏方向 | 2026 年 10 月 15 日\n8 × RTX 4090 / 4 × A800 80GB'),
    call('这一轮只保留一个优先验证的主候选：让单步视频学生在生成当前内容时，同时合成下一段可用的终态记忆，尝试省掉额外的完整缓存写入前向。'),
    p('工作名：**终态记忆蒸馏**。冻结已训练的因果视频生成器，仅学习轻量状态预测头；目标是在既有单步生成能力之上，把稳态的“1次生成 + 1次写缓存”变成“1次生成 + 小模块”。这是模型学习与推理计算图的改动，不以硬件、内核或量化为贡献。'),
    tb(['判断','本轮结论'],[
        ['优先主线','终态记忆蒸馏：计算删除对象明确，已有真实单步权重，可先小规模证伪'],
        ['探索副线','按学生更新影响训练critic：数学上有意义，但本轮反例显示其代理目标仍不够可靠'],
        ['上一版方案','critic输运降级；后续影响路由改作诊断工具；条件分叉覆盖继续保留为应用假设']
    ],[97,394]),
    p('主候选尚未证明有效。它比上一轮更值得先试的原因，是能够用短实验回答三个明确问题：真的有一遍可省的计算吗？直接复用是否确有质量缺口？小模块是否能补回这个缺口？'),
    small('本轮新增：多峰critic压力测试、注意力状态反例、加权DSM偏差检查、公开权重与源码入口核验。没有下载大模型权重或连接你的GPU集群；所有小实验与计划均分别标注。'),cover=True)

page('01 为什么调整上一版的排序',
    tb(['上一版候选','更深入检查后的发现','现在的定位'],[
        ['预测式critic输运','简单两快照外推也具有同样的一阶预测精度；输运涉及导数误差和额外学习成本。','先降级；数学成立不代表方法有竞争力'],
        ['后续影响驱动的步数分配','LeMiCa / OnlineCache已优化整体质量；D3P / POGP已有基于任务价值的去噪预算。[14]-[17]','可做视频诊断；宽泛机制不够新'],
        ['固定历史分叉覆盖','同历史多样本energy蒸馏在气象已有同构目标。[18]','需先证明视频特有问题'],
        ['新的终态记忆蒸馏','不是从零蒸馏更强生成器，而是替换明确的完整状态写入调用。','当前最值得先验证；仍需强近邻对照']
    ],[123,238,130]),
    h('判断依据从“有道理”提高到“能打败什么”'),
    p('原方案的解析高斯例子只能说明一条公式正确。这一轮改为：对照简单但强的方法，注入实际估计会遇到的误差，再观察最终学生更新，而不是只看中间score指标。对新候选也采用同样标准，若直接复用已足够好，就没有必要训练预测头。'),
    h('不靠换名称获得新颖性'),
    p('缓存预测、行为蒸馏、output + LSE匹配和on-policy数据收集分别都有先例。本报告不把这些单项当原创。主候选要证明的是：在冻结的单步视频学生上，已生成的终点与同次前向特征足以低成本构造兼容的后续记忆，而直接复用和一般KV回归做不到。'),
    small('“未发现完全同款”只是一轮定向查新的结果。是否达到投稿级贡献，取决于真实模型上的诊断与受控实验；不能由报告文字保证。'))

page('02 主候选：单步生成，还有一遍隐藏计算',
    p('因果视频模型把历史内容写入每层的key / value状态，供后面的latent块使用。某些公开配方先从噪声生成当前块，再把生成结果放回同一模型，以干净时间条件做一次完整前向，仅用于提交历史状态。[01][02]'),
    f('基线：x0_n = G_theta(z_n, S_<n)\n      S_n = Write_theta(x0_n, clean_time, S_<n)\n候选：x0_n, h_n = G_theta(z_n, S_hat_<n)\n      S_hat_n = Head_psi(h_n, x0_n, time, position)'),
    h('最值得尝试的是单步稳态'),
    p('选定的公开framewise模型，首个latent块仍使用4次去噪；后续块只使用1次去噪，但还额外写一次干净缓存。我们先保持首块流程完全一致，只替换后续块的写入。[02][03] “framewise”指latent帧组织，不等于每个RGB帧单独调用模型。'),
    p('若去噪与干净写入耗时接近，稳态模型成本可从2次完整调用降为1次调用加预测头。但这不是2倍端到端加速保证：首块、文本编码、VAE、状态存取和预测头都会占时。必须分别计时。'),
    h('为什么它仍有研究问题'),
    p('最后去噪调用中的KV来自带噪输入；干净写入中的KV来自已经生成的终点。两者可产生相同或相近的当前画面，却对之后的视频产生不同影响。直接复用已在Causal-rCM和FlashForward中出现，所以剩余问题是：能否在不重训原reader、不使用未来锚点的情况下，低成本补上这段状态差异。[04][06]'),
    call('核心假设：终点信息已经产生，重新运行完整DiT来编码它可能存在可学习的冗余；但“信息存在”不等于小模块一定能提取，必须测容量与闭环误差。'))

page('03 已核实的起点：无需先训练单步大模型',
    tb(['资产','本轮核验内容'],[
        ['模型','zhuhz22/Causal-Forcing中的causal-forcing++/framewise-1step.pt'],
        ['代码固定版本','8202903297918bcd12810fc4230a4db7fe065e64'],
        ['权重仓库固定版本','2f8eb8bb6eeb1238da9d13e5420d342a74d634a6'],
        ['采样配置','后续denoising_step_list=[1000]；首块[1000,750,500,250]；num_frame_per_block=1'],
        ['需要替换的操作','pipeline/causal_inference.py中生成后对clean / context_noise输入的额外generator调用'],
        ['可用性证据','仓库元数据与权重HEAD返回成功；文件约5.68GB。未下载，未在GPU加载或运行。']
    ],[115,376]),
    p('仓库标注为单步的文件和推理入口已经找到，具体模型质量仍需先复现。公开资源可用性与运行成功是两回事。固定代码版本能避免之后主分支变化造成NFE或配置理解偏差。[01]-[03]'),
    h('第一次跑模型先做四个对照'),
    p('①原样clean写入；②不训练任何东西，直接复用最后一次去噪KV；③便宜的逐层仿射KV校准；④同容量但只使用终点latent的预测头。随后才加入“终点 + 去噪特征”的完整设计。这样可以判断收益来自哪里，而不是只比较删掉重要步骤的坏基线。'),
    small('入口已核实不等于本报告完成了实现。完整adapter仍需修改模型forward以暴露合适的层特征，并保留正确的归一化、RoPE、cache位置和写入边界。'))

page('04 模型怎么改：终点与去噪特征联合写状态',
    p('第一版冻结G_theta。对第l层取最后去噪时、已归一化并完成时间调制的attention输入h_l，以及当前终点的patch embedding e(x0)。小型残差头预测clean-context的key / value修正；时间与位置条件沿用原模型。'),
    f('(Delta k_pre_l, Delta v_l) = B_l act(A_l [h_l;e(x0)]+b_l)\nK_hat_cache_l = RoPE(k_last_pre_l + Delta k_pre_l)\nV_hat_cache_l = v_last_l + Delta v_l'),
    p('保持原token数量和历史窗口，不同时做压缩、量化、动态稀疏或新日程。这样唯一需要归因的变化就是：完整的clean写入被预测头替代。若做残差而不是绝对预测，初始化为零就退化到直接复用基线。'),
    h('目标状态用同一个冻结模型产生'),
    p('教师不是另一个14B视频模型，而是同一G_theta在当前生成终点上执行一次真实clean写入。训练目标S*_n = Write_theta(x0_n, S_hat_<n)：与预测头看到的既有历史相同，只对当前待提交块做反事实clean写入。避免拿完全不同历史的“理想状态”当不可达标签。'),
    h('关键接口细节'),
    p('k_last_pre是norm_k(k(h_l))的输出，处于pre-RoPE坐标。原cache中的key已经旋转，不能代入上式。V的布局和索引保留，数值由头预测；历史槽位不变。训练用只读历史与可微当前KV拼接，避免原地写入破坏梯度。'),
    p('保留各层hidden的临时显存也计入成本，可先比较已有KV输入与hidden输入。逐token小头缺少直接的跨patch终点交互，能否从已有hidden补足是待证假设；不能靠无限加空间层，把头变成另一个完整writer。'),
    small('参数量只是设计估算：输入2d、低秩r、输出2d、L层，主要线性参数约4Ldr。若L=30、d=1536、r=16，则约2.95M，另加偏置 / 条件参数；实际维度和耗时以代码profile为准。'))

page('05 损失必须约束缓存被如何使用',
    p('直接匹配KV张量是合理的warm-up和强基线，但不同误差方向对未来注意力影响不同。我们复用已有行为蒸馏思想：在未来查询q上匹配局部注意力输出o和log-sum-exp量ell。[07]-[09]'),
    f('o(q;K,V) = softmax(q K^T / sqrt(d_h)) V\nell(q;K) = log sum_j exp(q k_j / sqrt(d_h))\nL_state = E_q[ ||o_hat-o*||^2 + beta * (ell_hat-ell*)^2 ]'),
    h('为什么不能只匹配o'),
    p('待写入块会与其他历史和当前token竞争注意力。仅匹配该块内部归一化输出，无法保证分配给该块的总权重正确。output与LSE共同确定该块参与合并注意力时的贡献；这是KVSculpt已有的契约，不是本报告的新定理。[07]'),
    f('o_full = ( exp(ell_C) o_C + exp(ell_R) o_R )\n         / ( exp(ell_C) + exp(ell_R) )'),
    h('还需一层完整reader约束'),
    p('局部attention匹配不能保证深层query不变。下一块的同一噪声输入x_t分别读取预测状态与真实clean状态：L_reader = ||G_theta(x_t,S_hat)-stopgrad(G_theta(x_t,S*))||²。reader权重冻结；预测分支保留对当前cache的梯度，教师分支用no_grad。'),
    p('先用状态loss训练头，再少量加入reader约束；不要立刻叠加DMD、GAN和奖励模型。主问题是状态兼容性，额外损失会让归因失去清晰度。beta等尺度在独立验证集固定。'),
    small('d_h为单个head维度，此模型为128，与参数估算的model width 1536不同。未来query仅作训练监督，推理时不读取未来帧、动作或teacher轨迹。'))

page('06 本轮检查：两个容易漏掉的状态错误',
    img('attention_checks.png',239.4),
    h('反例一：局部输出loss为零，完整输出仍明显改变'),
    p('缓存块两条value为0和2，基准logit相同；另一个历史块value为0、总权重为2。给缓存块所有logit加log(9)，其内部输出仍为1，output-only loss仍为0，但合并后的输出从0.5变成0.9。LSE差log(9)能发现问题。'),
    h('反例二：训练query全匹配，未见query仍失败'),
    p('在二维key上只沿训练query的正交方向改变预测key，可以使训练时output和LSE误差都为0；改用另一个正交query后，合并输出误差为0.4。因此，单次teacher-query拟合结果不能代替闭环rollout验证。'),
    call('这两个反例不证明预测头一定有效；它们说明哪些看似成功的训练指标不足以支持方法。状态loss必须同时检查注意力总质量和真实未来query上的泛化。'),
    small('精确CPU计算，未训练视频模型。复核脚本attention_checks.py与结果JSON一并保留；output + LSE思想明确来自[07]。'))

page('07 闭环训练：让预测头看到自己的历史',
    tb(['阶段','如何收集 / 优化','要避免的问题'],[
        ['1 离线warm-up','原模型生成短视频；记录最后去噪特征、终点及真实clean KV。','只看teacher历史可能低估实际状态误差'],
        ['2 混合历史','逐步以预测KV替代之前的状态；当前块目标用同一历史下的真实clean写入。','预测输入与教师目标必须在相同可达历史上'],
        ['3 闭环查询','主要采下一块实际单步时间（warp后）的query，并加入不同未来seed分支。','额外时间点仅属覆盖诊断，不称on-policy'],
        ['4 冻结reader校准','少量更新加L_reader；先只反传一个未来块，验证后再考虑两块。','没有必要保存分钟级完整反传图'],
        ['5 闭环验证','先测原生约5秒；统一窗口扩展验证后，再测20/60秒。','长时属额外分布外压力测试，不是现成入口能力']
    ],[82,212,197]),
    h('两项必要的工程检查属于模型正确性'),
    p('collector用no_grad；历史、终点、层特征、局部loss的q及未来x_t均detach。当前头输出保留梯度，L_reader只经当前预测cache回传。标签使用独立的KV与索引副本，不写回学生历史。另建训练入口：原推理脚本关闭了全局梯度。'),
    p('固定配置默认21个latent帧，KV容量32760 tokens，局部窗口未启用。长视频须先验证滚动窗口和位置处理；clean、reuse、head共用同一扩展，避免把窗口变更误差归因给预测头。'),
    h('收窄论文主张'),
    p('当前承诺范围是已有clean-context checkpoint的兼容状态合成。不是保证跨所有backbone、长短视频、条件控制和采样步数通用。先冻结reader检验状态建模是否成立，再把可选LoRA共同适配当另一项实验。'),
    small('on-policy收集与短程reader loss都属于已有训练工具。新贡献若成立，应体现在终态信息如何让轻量writer替代完整写入，以及对应的质量-耗时前沿。'))

page('08 必须正面对比的最近邻',
    tb(['近邻','已经做了什么','本候选必须区分的部分'],[
        ['Causal-rCM [04][05]','直接复用最后去噪KV，并训练匹配的noisy context。','不声称首次N+1→N；冻结clean-reader下预测终态状态'],
        ['FlashForward [06]','in-flight KV复用；辅以稀疏clean anchors及stage pipeline。','不依赖未来anchor；实验用相同设备数与非pipeline口径'],
        ['KV Prediction / SemPIC [08][09]','预测KV或用行为蒸馏训练Writer。','这些原理非新；新增对象是同次视频生成的终点和continuation状态'],
        ['KVSculpt [07]','attention output + LSE以及cache蒸馏。','直接继承loss契约，不包装成新统计目标'],
        ['SGF+ [10]','分离writer / denoiser梯度，仍重新编码生成的clean latent。','明确删掉一遍完整writer计算，而非只改善写入训练'],
        ['Separable Causality [11]','较重causal encoder与轻denoiser的职责分离。','复用既有单步学生；不重新设计整套生成架构'],
        ['CORAL-KV / UnStep [12][13]','跨层KV预测压缩；或复用clean-pass做修复。','前者保留原KV来源，后者仍可需要clean调用；与目标成本不同']
    ],[125,178,188]),
    p('最危险的竞争者是Causal-rCM的直接复用。其某些两步设置已报告很小的质量差异。这意味着两步场景未必有足够改进空间，因此本轮把主验证对象收窄到单步稳态；公开表中的不同训练预算结果只能作动机，不能当受控因果证据。'),
    small('最终若简单仿射校准或直接复用同样好，完整方案即无独立价值。报告不以方法模块更多或loss更复杂作为优点。'))

page('09 最小实验：用三道门决定是否立项',
    tb(['门槛','最小实验','继续 / 停止标准'],[
        ['门1：计算与缺口','32个未见分层prompt×2seed；固定checkpoint，比较clean / reuse / affine。','affine仅在训练prompt拟合；确认实际成本和可重复质量缺口，否则停止'],
        ['门2：可预测性','训练rank8/16/32小头；比较只读最后KV、只读终点、两者联合；用独立prompt。','恢复reader误差和短视频质量，而非只压低训练KV MSE'],
        ['门3：闭环与实速','原生约5秒；窗口验证后再20/60秒；3训练seed；计全部头开销。','质量达预定容忍度，端到端耗时确实改善，且胜过同成本简单方法']
    ],[83,228,180]),
    h('主消融只保留能回答问题的项'),
    p('输入：去掉终点信息；去掉层特征。目标：raw KV回归、output-only、output+LSE、再加reader一致性。历史：仅clean收集与闭环收集。容量：rank8 / 16 / 32。模型：冻结reader与可选同预算LoRA。各组沿用相同数据和采样日程，训练预算一致。'),
    p('评价覆盖身份一致性、背景布局、运动平滑度、动态程度、闪烁与指令遵循。更静止的视频可能掩盖状态损伤，运动与静态prompt应分别报告。'),
    p('正式质量-耗时曲线还应加入同runtime的UnStep及原生两步checkpoint。UnStep若启用clean-pass终点修复，不能删掉该调用后仍声称保持原输出；各方案按真实调用与总耗时比较。[13]'),
    call('若只能靠频繁完整clean刷新维持质量，应报告刷新次数后的真实收益；不能仍称每块一遍。若头越来越接近完整writer成本，证明小模块假设失败，不继续无上限加容量。'),
    small('以上样本量用于筛选，不是统计功效保证。超参数在验证集确定；最终使用完整官方评测及按prompt聚类的置信区间，训练seed波动单独报告。'))

page('10 计算账本与用户资源的匹配',
    p('相比完整DMD重新训练，此主线只需一个冻结的1.3B模型和小型writer头；标签由同一模型产生，不必同时放学生、14B教师和可训练fake三套网络。训练时反传reader输入仍需要激活，不能把冻结权重误当作零显存。'),
    f('稳态基线成本 = F_denoise + F_clean + C_other\n候选成本 = F_denoise + H_head + C_other\n加速比 = (F_denoise + F_clean + C_other)\n        / (F_denoise + H_head + C_other)'),
    p('仅作示例：令F_denoise=F_clean=1，H_head=0.05。若其他成本为0.2，理想比为2.2/1.25=1.76；若其他成本为1，则为3/2.05=1.46。0.05不是测量值，实际头成本、cache写入与VAE必须profile。'),
    tb(['资源','建议用途'],[
        ['4 × A800 80GB','冻结reader的局部反传、预测头训练、短闭环实验；独立采样与数据并行即可起步'],
        ['8 × 4090','并行prompt / seed采样、离线标签和评价；先确认单卡权重、文本编码器和KV能放下'],
        ['开始时先做','预编码文本；逐层 / 逐chunk收集；先保存有限特征避免海量全KV磁盘数据'],
        ['成本分账','数据生成、真实clean标签、head训练、reader校准、评测分别计GPU小时']
    ],[116,375]),
    p('计时先预热并同步，正式P95建议至少100次。首块4次去噪与clean写入都计总成本；另报内容就绪TTFC、含状态准备的首块耗时及整段E2E。原仓库首块timer不含clean刷新，生成timer不含VAE，不能直接当端到端指标。'),
    small('本次未测你的机器；所有时间与显存上限均需pilot确认。显卡资源是落地约束，研究贡献仍来自状态预测模型和删除的一次神经网络前向。'))

page('11 探索副线：critic误差要按学生更新衡量',
    p('DMD里并不是所有score误差都会等量影响学生。若e=s_fake-s_true，J=partial x_t / partial theta，那么污染参数更新的是E[J^T e]。因此可研究：有限critic容量应该优先服务哪些方向，而不是只最小化全空间score MSE。'),
    f('b = E[J^T e]\n||P^(1/2) b||^2 <= E[e^T M(x_t,t,c) e]\nM(x_t,t,c) = E[J P J^T | x_t,t,c]'),
    p('P为预先固定的正半定度量；对照锁定参数化与P，本式不自动具备任意重参数化不变性。LoRA应在有效权重切空间定义扰动。可用u~N(0,P)的Ju估计条件二阶矩，再拟合M；第一版只用受限参数子空间，并加tau I保持正定。'),
    h('有意义的改变是估计任务，不是最后乘一个投影'),
    p('若把输出score梯度投影到J的列空间，再通过J反传，J^T Projection_J = J^T，结果不变。因此简单“切向投影DMD”是代数冗余。可争取的贡献只能来自critic有限预算的训练目标和表示变化。'),
    h('一个可保持正确极值的形式'),
    f('L_fake = E[ (s_phi(x_t) + epsilon/sigma)^T\n             stopgrad(M_omega(x_t,t,c))\n             (s_phi(x_t) + epsilon/sigma) ]'),
    p('权重只依赖noisy query及明确条件、保持正定并固定时，population最优score不变。该性质不代表有限网络下平均参数梯度偏差一定更小；下一页的反例说明这个区别很关键。'),
    small('Hodge可观测误差[19]研究扩散边际动力学，不能直接当作上述学生参数梯度的定理。加权score matching和Sobolev正则[20]本身也不是新技术；当前副线尚不足以主推。'))

page('12 副线的两个反例：先修目标，再谈视频',
    h('直接用同一训练样本的Jacobian加权，会改变目标'),
    p('设G_theta(z)=theta z，在theta=1时x=z+epsilon，z和epsilon独立标准正态，真score为-x/2。若把同一对(z,epsilon)的J²=z²直接乘DSM，即使stop-gradient，该权重仍含隐藏clean变量，最优解发生偏移。'),
    f('s_true(x) = -x/2\ns_naive(x) = x(2-x^2) / [2(2+x^2)]\nx=1:  s_true=-1/2,  s_naive=+1/6'),
    p('安全做法先拟合M(x)=E[z²|x]=1/2+x²/4，再把只依赖x的权重用于DSM。停止梯度只影响微分路径，不会消除概率条件依赖。'),
    h('即使权重完全正确，也不保证平均更新偏差下降'),
    p('对受限critic族s=a(x+0.25x³)进行population最优拟合，本轮解析积分得到：条件影响权重降低了逐样本参数梯度误差二阶矩，却使平均偏差平方变大。另一个tanh族两项都改善，说明不能只挑有利critic族展示。'),
    tb(['指标','普通DSM','条件影响权重'],[
        ['逐样本参数梯度误差二阶矩','0.25598','0.18750'],
        ['平均参数梯度偏差平方','0.009365','0.035156']
    ],[245,123,123]),
    call('结论：这条副线更适合作为“critic评价与优化目标是否错位”的基础研究问题。先测平均bias、有限batch方差和真实更新后的KL，而不是把一个上界降低写成训练一定变好。'),
    small('本页为1D解析例子，不是神经网络或视频实验。需要在多峰分布、有限样本学得M及真实低预算critic上继续验证。当前优先级低于终态记忆蒸馏。'))

page('13 旧critic输运：压力测试改变了判断',
    p('上一版用高斯分布验证score输运公式。本轮加入多峰、低噪声、训练漂移和旧critic估计误差；对照有stale、EMA、两快照外推、oracle输运及受限仿射输运。这些实验用于筛除过强论断。'),
    h('简单外推本来就是强对照'),
    f('s_extrap(u+Delta) = 2 s(u) - s(u-Delta)'),
    p('在平滑训练轨迹上，它同样达到O(Delta²)的局部误差阶数。因此，“输运能把误差从一阶降到二阶”不足以支持增加漂移学习器和输入VJP。必须比较等信息、等成本条件下的实际误差与生成收益。'),
    h('旧critic导数噪声可能被放大'),
    f('e_pred ~= e - Delta * ( v_prime * e + v * e_prime )'),
    p('常速度平移时，误差e=a sin(omega x)经显式Euler修正，幅度放大系数为sqrt(1+(Delta*v*omega)^2)。一阶预测可能放大高频估计噪声；最终学生梯度还取决于积分抵消与batch方差，不能只凭score RMSE下结论。'),
    call('新的定位：不再作为优先大模型主线。只有当真实旧critic的可预测漂移占主导，且低成本实现稳定超过简单外推 / 额外DSM，才值得恢复优先级。'),
    small('本轮详细解析与学得critic的结果见下一页和复核数据。oracle漂移 / 真实score是诊断上界，不是可部署方法；不同对照所用快照和导数信息也需单独计成本。'))

page('13 附：critic压力测试的实质结果',
    p('以下是CPU多峰解析与RBF回归critic实验。没有训练真实两步生成器，也没有等GPU小时比较；目的在于检查原先的推荐是否有充分依据。'),
    img('critic_round2_summary.png',360.1),
    tb(['学得critic设置','简单外推','oracle输运','当前分布重新拟合'],[
        ['48个精确训练期望设置','0.185','0.200','0.177'],
        ['240个有限DSM样本设置','0.439','0.445','0.429']
    ],[198,91,91,111]),
    p('表为batch=32时参数梯度误差RMSE相对各自stale基线的比值中位数，越低越好。由精确一、二阶矩计算，非随机优化器的训练成绩。完整预设设置均纳入，未按胜负筛选。'),
    small('公平性边界：输运获知精确条件漂移，外推获知两次历史score及等间距平滑轨迹；fresh RBF是完整重拟合，不是一次DSM更新。接近目标的32个偏差设置中，外推 /输运各12次令真实KL增大；这只否定无条件改善，不等于它们总会失败。'))

page('14 进一步排除的“合理但不够新”方案',
    tb(['方案','本轮新增的近邻 / 问题','处理'],[
        ['分开CFG正负分支保条件能力','PDM / NBA已有正向预测与条件方向匹配。[21]','不作为新主线'],
        ['两步采用不同参数并端到端训练','Z-Image Turbo++已有step-decoupled参数与迭代正则。[22]','不能直接复用为新idea'],
        ['重新拟合guided sampler的参考score','GuidanceNFT已处理guided样本与reference速度失配。[23]','迁移到DMD需额外独立贡献'],
        ['只匹配未来attention输出','遗漏分块attention总质量；本轮精确反例显示失效。','必须匹配LSE并做reader验证'],
        ['只按学生Jacobian大小加权DSM','隐藏clean变量引入目标偏差；stopgrad不能修复。','只允许query条件化的权重'],
        ['把很多模块一起训练','无法识别改进来自状态建模、更多容量还是额外预算。','冻结reader与逐项对照起步']
    ],[158,229,104]),
    h('目前最清楚的论文问题陈述'),
    p('“少步视频模型已能预测当前内容，但历史写入仍重复完整推理。我们研究在冻结生成器条件下，能否利用生成终点与同次去噪表征，学习足以驱动后续生成的记忆状态，从而避免这次重复计算。”'),
    p('如果成立，结果应有三部分：单步状态差异的诊断、轻量状态合成与闭环训练、相同runtime下的质量-延迟改善。单独给出小头和几段好看的视频，不足以完成论证。'))

page('15 开始实验时，只需要回答这五个问题',
    tb(['顺序','问题','形成的证据'],[
        ['1','选定checkpoint真的每块多跑一遍吗？','源码调用计数 + GPU profile；区分首块与稳态'],
        ['2','直接复用最后KV比clean写入差多少？','同checkpoint、prompt、seed、日程的短 / 长视频成对结果'],
        ['3','小模块是否读懂生成终点？','endpoint / hidden输入消融，独立query与reader误差'],
        ['4','状态错误是否越滚越大？','预测历史上的闭环实验、动作 / 场景转变与长时漂移'],
        ['5','质量相当时端到端确实更快吗？','全部额外头开销、刷新调用、VAE和首块都计入']
    ],[47,239,205]),
    h('资源投入顺序'),
    p('先用少量prompt完成门1，再训练rank8/16小头。通过之后才扩大prompt和视频长度；最后做三seed、第二个checkpoint与可选共同微调。先证明“删一遍”有价值，不急于把训练做成完整DMD大工程。'),
    h('关于这轮结论的把握'),
    p('相较第一版，当前主线的计算目标、已有资源、最近邻和否决条件更明确；但尚没有真实模型实验，不能称已经验证高质量idea。另一条critic副线保留数学兴趣，暂不给它大规模训练优先级。'),
    call('本轮最重要的改进是收窄：一个可直接验证的模型主线，配足够强的否决实验；不再同时把几条尚缺实证的直觉都列为高优先级。'),
    small('本文与第一版互为补充；涉及排序时以本轮判断为准。所有公开代码、模型revision及小实验文件建议在后续实验仓库固定，避免结果口径漂移。'))

refs={
 '01':{'title':'Causal Forcing++: Scalable Few-Step Autoregressive Diffusion Distillation for Real-Time Interactive Video Generation','url':'https://arxiv.org/abs/2605.15141','meta':'公开少步因果学生与推理配方。'},
 '02':{'title':'Causal-Forcing: causal_inference.py, pinned source','url':'https://github.com/thu-ml/Causal-Forcing/blob/8202903297918bcd12810fc4230a4db7fe065e64/pipeline/causal_inference.py','meta':'固定代码版本；额外clean-cache调用、首块日程与latent块组织。'},
 '03':{'title':'Causal-Forcing public one-step checkpoint','url':'https://huggingface.co/zhuhz22/Causal-Forcing/blob/2f8eb8bb6eeb1238da9d13e5420d342a74d634a6/causal-forcing%2B%2B/framewise-1step.pt','meta':'权重入口核实；HEAD /元数据成功，不代表已下载或运行。'},
 '04':{'title':'Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation','url':'https://arxiv.org/abs/2606.25473','meta':'§3.1.3已有最后去噪KV复用；不能宣称首次删cache-only调用。'},
 '05':{'title':'NVlabs rcm: Causal_rCM.md','url':'https://github.com/NVlabs/rcm/blob/main/Causal_rCM.md','meta':'noisy context、1/2/4步日程与checkpoint说明；实际实验需固定版本。'},
 '06':{'title':'In-Flight KV Cache with Clean Anchors for Faster Autoregressive Video Diffusion','url':'https://arxiv.org/abs/2609.32540','meta':'FlashForward；noisy KV、clean anchors与pipeline的强近邻。'},
 '07':{'title':'KVSculpt: KV Cache Compression as Distillation','url':'https://arxiv.org/abs/2603.27819','meta':'§3.2 output + LSE契约。本报告使用此思想，不声称原创。'},
 '08':{'title':'SemPIC: Learning Semantic Position-Independent KV Caches','url':'https://arxiv.org/abs/2607.28069','meta':'Writer行为蒸馏与冻结Reader的跨领域近邻。'},
 '09':{'title':'KV Prediction for Improved Time to First Token','url':'https://arxiv.org/abs/2410.08391','meta':'KV预测的已有工作；不能主张首次预测缓存。'},
 '10':{'title':'SGF+: Decoupling Gradient Flows for Autoregressive Video Generation','url':'https://arxiv.org/abs/2610.10429','meta':'writer / denoiser职责和梯度分离，2026-10预印本。'},
 '11':{'title':'Causality in Video Diffusers is Separable from Denoising','url':'https://arxiv.org/abs/2602.10095','meta':'重causal encoder与轻denoiser分工，模型结构近邻。'},
 '12':{'title':'CORAL-KV: Compact Cross-Layer Prediction and Residual Coding for Autoregressive Video KV Caches','url':'https://arxiv.org/abs/2610.12514','meta':'2026-10预印本；KV跨层预测与压缩的区别。'},
 '13':{'title':'UnStep: Training-Free Acceleration of Causal Video Diffusion with Fewer Steps Than Distillation','url':'https://arxiv.org/abs/2609.32518','meta':'复用clean-pass修复以及更少采样步；总体收益含runtime优化。'},
 '14':{'title':'LeMiCa: Lexicographic Minimax Path Caching for Efficient Diffusion-Based Video Generation','url':'https://arxiv.org/abs/2511.00090','meta':'整体路径误差与缓存日程，不只优化local heuristic。'},
 '15':{'title':'OnlineCache: Learning Dynamic Caching Policies with Error Correction for Efficient Diffusion Inference','url':'https://arxiv.org/abs/2607.29398','meta':'全局质量目标下的动态预算策略与学习校正。'},
 '16':{'title':'D3P: Dynamic Denoising Diffusion Policy via Reinforcement Learning','url':'https://arxiv.org/abs/2508.06804','meta':'按动作对任务成功的重要性分配去噪步数。'},
 '17':{'title':'Learning When to Stop: Prefix-Optimal Dynamic Diffusion Policies for Continuous Control','url':'https://arxiv.org/abs/2608.05084','meta':'POGP；Bellman式prefix value与动态停止。'},
 '18':{'title':'Diffusion Distillation for Efficient Weather Ensembles','url':'https://arxiv.org/abs/2608.27728','meta':'同历史条件多样本energy蒸馏的实质先例。'},
 '19':{'title':'Diffusion Models Observe Only Gradients: A Geometric Perspective on Score Matching Errors','url':'https://arxiv.org/abs/2606.06179','meta':'Hodge / Fokker-Planck可观测性不等于学生参数梯度可观测性。'},
 '20':{'title':'Sobolev Regularized Score Difference Estimation in Diffusion Models','url':'https://arxiv.org/abs/2608.18237','meta':'score difference估计及小样本稳定性近邻。'},
 '21':{'title':'Rethinking Classifier-Free Guidance in On-Policy Diffusion Distillation','url':'https://arxiv.org/abs/2607.24731','meta':'NBA与Positive-Direction Matching；排除泛化CFG分支方案。'},
 '22':{'title':'High-Fidelity Two-Step Image Generation via Teacher-Aligned End-to-End Distillation','url':'https://arxiv.org/abs/2606.12575','meta':'Z-Image Turbo++，step-decoupled参数化与端到端迭代正则。'},
 '23':{'title':'GuidanceNFT: NFT-Based Reward Fine-Tuning of Guided Diffusion Models','url':'https://arxiv.org/abs/2610.15856','meta':'2026-10-11预印本；guided样本参考速度失配校正。'},
 '24':{'title':'SGMD: Score Gradient Matching Distillation for Few-Step Video Diffusion Distillation','url':'https://arxiv.org/abs/2605.30116','meta':'critic / student耦合更新的强对照；副线需超过等预算既有方法。'}
}

def write():
    for n in range(3):
        page(f'参考来源 {n+1} / 3',{'type':'refs','keys':[f'{k:02d}' for k in range(8*n+1,8*n+9)]},
             small('检索截止2026-10-15。外部文献支持已有事实；算法设计、优先级和停止门槛属于本报告研究判断。小型CPU实验不替代真实模型验证。'))
    (ROOT/'report_content.json').write_text(json.dumps({'pages':pages,'refs':refs},ensure_ascii=False,indent=2),encoding='utf-8')
    print('logical_pages',len(pages))

if __name__=='__main__':write()
