from pathlib import Path
import json,html,hashlib,re
from html.parser import HTMLParser
root=Path.cwd();base=root/'output/sdxl-crossdistill';out=base/'project_audit_20261112'
rows=json.loads((out/'experiment_results.json').read_text(encoding='utf-8'))
sections=[]
def section(title,body):sections.append((title,body))
def link(path,label=None):
 return f'<a href="{(root/path).as_uri()}">{html.escape(label or path)}</a>'
def table(head,rs):return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rs)+'</tbody></table></div>'
section('审计结论与证据边界', '''<p>这条项目线已经建立了可运行的SDXL四步蒸馏与评测流程，形成17组自训G160的历史COCO10K结果、2个公开权重参照，以及3组已训练完成并于11月11日启动补测的模型。最强的可复核现象是：<b>PCM＋仅末步学生DM显著降低本项目FID和中心Patch FID，但CLIP1024同时下降；GAN对局部指标有改善，尚不足以证明综合质量提升。</b></p>
<p>核心边界加噪、fake条件回归目标、梯度隔离和微批累积的21项CPU测试通过。发现1项可复现的队列验收错误，已经在本地作简洁修正。另有若干高影响设计问题，主要涉及低噪声fake降权、PCM系数未经标定地沿用、计算预算和监督覆盖同时变化、以及fake跟踪诊断不足。它们影响因果解释，不能统称为“训练代码崩了”。</p>
<p><b>本次实时限制</b>：11月12日两次连接172.18.32.78均SSH超时，未取得新三组的最终分数。报告使用本地保存的评分JSON和11月11日启动证据；最新补测状态以“待核验”标记。当前生产环境也未重新验证源码哈希。本次没有启动新训练或修改远端正在运行的任务。</p>
<p>审计覆盖20个关键源码文件及主要配置/实验档案。它是从反方角度复核已有实现的一次单审计者检查，不把早期审计通过当作当前代码正确性的充分证据。CPU测试不能替代全尺寸多卡FSDP或真实SDXL长训验证。</p>''')
section('项目结构与实验状态', table(['位置','职责','当前使用建议'],[
['sdxl_parallel / sdxl_uniform4_20261022','旧clean目标、DINO＋RGB局部GAN、分批生成和反向','保留为历史实验解释依据，不能直接代替新版边界训练器'],
['sdxl_research','边界DM、PCM、阶段条件fake、采样先验、参数混合、共享训练工具','审计主入口；实际已跑实验仍以各run/source快照为准'],
['tmp/prepare_*、run_*','每项实验的隔离副本、修改和启动队列','不少改动仅在这些脚本中，目录名不能作为配置证据'],
['output/sdxl-crossdistill','报告、局部日志、评分、配置、清理收据','本报告的experiment_results.json建立统一结果索引'],
['missing_evaluation_20261111/code','三组COCO10K补测入口','固定GPU UUID、逐文件哈希、固定10000项、断点特征复用'],
['pcm_dm4_weight_ablation_20261109','PCM＋末步DM系数1/2的成对代码','已通过CPU验证，未发现本线程内实际GPU训练完成证据']])+
'''<p>各组通常从原始SDXL独立初始化，数据是同一LAION衍生10K缓存。G160代表160次学生更新，batch64对应10,240次学生样本使用，约1.024遍10K训练样本。fake每G两次时是20,480次样本使用，三次时为30,720次。它们不等同于160个epoch。</p>'''+table(['实验状态','已知情况'],[
['历史完整COCO10K','下表17组自训及2个公开模型。不同样本规模的CLIP单独标注'],
['A×10＋1%参数混合','G160已完成，权重实验室备份并核验，11月11日首组补测已处理128/10000；11月12日结果待连接后核实'],
['局部边界归一化转向 / 原始转角','11月11日在.32.81核实均G160/F320且4分片齐全；此前“原始转角至少G105”的旧报告已过时'],
['低噪声30%组','旧队列曾因7GiB空间门槛退出；恢复中段fake、磁盘等待和本次验收修正目前只有本地证据，不能列为已训练'],
['H20直接末步GAN','完整更新止于G33，G34中断，无最终模型；已取消的完整COCO评测不自动重启'],
['D配额变体 / 无先验局部边界','分别G9、G6部分运行，不能与G160并列当作完整消融'],
['PCM＋末步DM等时延长','G160权重及完整评分有效；G171后退出，等时端点缺失'],
['20K数据','20,000训练＋32留出已准备；未发现已完成的20K正式训练结果']]))
section('完整结果台账', '''<p>FID与Patch FID均越低越好。Patch FID使用原生图像中心299×299裁剪。CLIP列为OpenAI ViT-B/32余弦×100，方括号内为样本数；128与1024的均值不可直接排统一名次。B/C的CLIP在总表中保留了三位小数。</p>'''+table(['实验','FID↓','Patch FID↓','CLIP↑ [n]','训练h'],[[x['name'],f"{x['FID']:.4f}",f"{x['Patch_FID']:.4f}",f"{x['CLIP']:.3f} [{x['CLIP_n']}]" if x['CLIP'] is not None else '未测/未汇入',f"{x['hours']:.2f}" if x['hours'] else '—'] for x in rows])+
'''<p>旧5组时间来自G0/G160固定图时间戳，后期组主要为完整更新跨度，均不含事后COCO评测。H20与A800、2卡与4卡之间的耗时不能直接解释为算法效率。单种子、G160预算下的小数点差异没有训练重复实验的不确定性估计。</p><img src="quality_tradeoffs.png" alt="历史FID与中心Patch FID关系图"><p>图中的左下方更好。公开模型的训练预算与自训实验不同。当前自训结果与DMD2仍有明显差距；高CLIP均值也不能抵消FID或纹理指标的劣化。</p>''')
section('四步训练实际计算了什么', '''<p>学生是一套共享SDXL UNet，时间条件区分四个推理位置。教师冻结，CFG8；fake是独立可训练UNet，跨区间版本额外带阶段embedding。文本编码读取缓存，VAE冻结。学生与fake全参数训练，约2.57B参数；单份BF16学生权重约4.78GiB。FP32主参数、梯度、Adam两份矩以及两套网络同时存在，解释了训练显存远大于单份权重。A800采用micro16×累积4、rollout8，仅学生Adam状态CPU卸载。</p>
<p>四个输入时间为999、749、499、249，终点用索引−1表示真正干净状态。每个G先更新fake，再累计学生监督，最后一次学生optimizer.step。中段前缀使用detach，后续损失不会沿整条四步轨迹反传。但四步共享参数，所以某一步更新仍会改变其他步的行为。</p>'''+table(['部分','真实操作','梯度与样本'],[
['首步PCM','真实训练latent加噪到随机高时间；教师推进两个10步短步；学生在推进前后预测共同749边界','目标分支stop-gradient；输入来自真实latent，不是学生第一步rollout结果'],
['中段DM','随机选第2或3步，匹配实际x499或x249；全选变体使用0.5DM2＋0.5DM3','仅选中局部调用有梯度；终点DM每G执行'],
['末步DM','从x249得到干净终点，再独立加噪，比较教师与fake方向','边界s=−1；teacher/fake方向和归一化均停止梯度'],
['fake基线','每G选中中段一次、末步一次，两个独立完整更新','中段从实际边界latent加噪；student输出detach；对应阶段条件一致'],
['1%混合','学生更新后，fake共享参数←0.99fake＋0.01student','fake专用embedding与Adam状态不混合；这是带fake自身优化的参数注入，不是普通推理EMA'],
['仅末步学生DM','删除学生中段loss，仍有PCM与DM4','既有最佳分数组仍训练中段fake，不能写成“全流程只训练末步”']])+
'''<pre>a = αₜ / αₛ
b² = σₜ² − a²σₛ²
xₜ = a·yₛ + b·ε
fake条件目标 = (σₜ / b)·ε
Lfake = mean[(b·εfake − σₜ·ε)² · min(1/b², 10)]

score = −εpred / σₜ
mT = (xₜ + b²·scoreT) / a
Z = mean(|yₛ − mT|), 下限1e−5
方向g = (b² / (a·Z)) · (scoreF − scoreT)
LDM = 0.5 · mean[(yₛ − stopgrad(yₛ − g))²]</pre>
<p>CPU解析高斯与有限差分测试支持该条件score目标和原始KL方向的符号、加噪Jacobian。实际代码还使用后验归一化和时间权重，因此训练显示的DM loss不是KL数值，也不应作为生成质量指标。</p>''')
section('采样先验与低噪声降权', '''<p>A/B/C跨区间支持分别是第2步509–979、第3步259–979、末步20–979。A在合法区间均匀抽样；B以30%抽900–979，以70%抽补集；C中段同B，末步把20–100与900–979的并集合计分配30%。C在末步的理论低噪声质量为0.3×81/161≈15.09%，高噪声约14.91%。这些配额是训练加噪分布，并没有改变四个推理时间点。</p>
<p>旧归一化转向和原始转角实验使用局部加噪区间509–739、259–489、20–239，与后来的跨区间A/B/C属于不同主线。先验形式也从连续曲线混合改成固定区间配额。不能把这些都叫同一个“DyMD先验”而忽略差别。</p>
<p><b>高影响设计问题：低噪声fake损失已经被额外降权。</b> 在干净终点b=σₜ，公式严格等价于 min(1,10σₜ²)×普通噪声MSE。20–100区间乘数均值约0.59984，t20约0.18686、t60约0.59126、t96起为1。该设置从10月25日边界DM实现引入，A/B/C及后续配方继承。引入理由是抑制中段小增量噪声的条件回归尺度，但没有单独实验支持把相同截断用于末步。</p>
<img src="fake_low_noise_weight.png" alt="末步fake低噪声损失乘数">
<p>“低噪声30%”仍是正确抽样比例，却不表示30%的损失或梯度贡献。仅在每个时间的误差相同这一假设下，0.3×0.59984与0.7的相对质量约为20.45%/79.55%；真实梯度占比还取决于网络误差和Jacobian。不能将20.45%宣称为实测梯度比例。</p>
<p>最新控制变量意图应落实为：学生PCM＋末步DM；fake恢复中段＋末步；末步student/fake同时改低20–100占30%、其余70%；中段保持C的分布，fake权重上限10保持一致。本次在本地构造该profile并验证中段分布逐项相同、末步低区间质量恰为0.3。远端尚无修正部署证据。</p>
<p>取消降权应另成一个明确消融。当前参数解析还把fake_track_pilot与fake_weight_max==10绑定，terminal_probe又写死10；需要同时解开入口限制并让探测读取真实配置，才能有效比较加权与未加权回归。直接改一个数值并绕过参数检查会破坏诊断的一致性。</p>''')
section('反方逐项审查：错误、风险与已有证据', table(['问题','证据与影响','建议及状态'],[
['P1 · 完成验收与实际日志不一致（确认错误）','新低噪声队列要求160条fake_student_mix；复制的基线训练器只记录heldout_fake_tracking。有效训练结束后也会触发断言失败。见修正前快照第54–55行','本地已改为复用verify_tracking_probes，检查配置、固定8样本、期望探测步数和混合率。正例/缺记录反例通过；未部署远端'],
['P1 · 新旧源码与队列状态脱节（操作风险）','低噪声组磁盘等待、恢复中段fake等修改保留在本地；远端曾部署fake末步两次版并因空间不足退出。目录存在或磁盘已释放不等于新方案已启动','启动前读取部署effective_config、源码哈希和进程，形成一次明确验收；本次不擅自恢复付费训练'],
['P2 · fake低噪声权重与采样干预耦合（设计风险）','末步回归被min(1,10σ²)缩放；A/B/C绝对fake loss不能横比来判断谁更跟得上','保留现有已跑组定义；将是否加权单列消融，并记录固定t未加权MSE'],
['P2 · PCM调试系数成为正式默认（历史设计缺口）','10月25日说明明确PCM=1为未校准调试起点；旧配方为10。新版结果后来沿用1，不能视作已优化','A1/A10是已完成的直接控制，结果呈FID、Patch、CLIP权衡；不以单项指标决定统一扩大PCM'],
['P2 · 全选DM的预算与fake覆盖混杂（实验设计）','DM234组fake从2变3，训练时间6.90→9.01h；同时改变学生覆盖和fake拟合预算','结论限于该组合。若要归因，固定fake预算或单列fake×3控制，并比较同G及同时间端点'],
['P2 · fake混合的“跟踪改善”证据不足（诊断）','探测仅8条、t60/499/940；混合前后对照只反映一次注入；原C无历史fake权重/留出探测可回补','保留“参数混合小幅指标变化”的描述；不能把本次before当原C的0%训练基线'],
['P2 · 中心Patch FID代表性有限（评测）','原生中心299仅覆盖1024图面积约8.5%；GAN训练随机局部crop或全图SAM，采样区域不同','保留当前主指标保证可比；之后可增加固定多位置crop和盲评，分别报告，不能称中心分数等于全图纹理'],
['P2 · 高频固定图的存储成本（资源）','每2G保存8图，单个G160可有约648张；多组累计6592张占9.24GiB','未来固定关键节点出图，其余只记标量。历史已按授权备份616张并清理，无需引入复杂存储框架'],
['P2 · 检查点“可测”与“可续训”混用（生命周期）','后期按用户节省空间要求只存BF16学生，fake/Adam/RNG缺失；全状态另有明确格式','记录checkpoint_kind。当前这些权重可推理，不能据此恢复同一优化轨迹'],
['P3 · 依赖源码和脚本散布（可维护性）','多个prepare脚本文本替换生成隔离训练器，主目录单个train.py无法代表全部实验','保留历史快照；维护一个小型实验索引，包含运行名、配置、source哈希、权重、评分和状态，避免新框架重写']])+
'''<p><b>没有列为现存错误的怀疑</b>：中段fake误用干净x₀、教师/fake梯度串入学生、累积batch64多除一次、局部反向损失遗漏、重复caption导致CLIP检索必然错记，以及旧版双卡按各自时钟进入集体存档等。相关代码或测试分别支持真实边界输入、梯度隔离、mean/accumulation与FSDP平均、循环yield后backward、等价caption检索处理及rank0广播控制。它们仍需区分历史版本，但不能为了“敢提问题”而把已修复或不可达路径写成当前P1。</p>
<p>还检查了“恢复时混合参数未核对”的嫌疑：当前正常参数入口禁止fake_track_pilot与resume同时启用，因此无法通过支持的入口复现所怀疑的静默混合率变更，本报告不将其列为已确认漏洞。</p>''')
section('GAN实现与结果能支持什么', '''<p>旧DINO＋RGB方案对干净终点做48×48 latent随机裁剪，VAE解码成384像素并取中心256，DINO分支再取中心224。四层DINO特征接可训练头，RGB分支含文本投影，冻结backbone仍保留对图像输入的导数。两支评分平均，hinge损失，D-only翻转，lazy R1每16次D更新执行。A×10配套GAN组每G更新D五次，共800次；学生仅G3、6…159接受GAN，共53次。</p>
<p>该组通过小图分块计算dGAN/dlatent，再把停止梯度的latent导数接回学生。实现保留了按chunk占有效微批比例的加权，能节约显存。VAE的GroupNorm使局部latent解码与全图同位置解码不完全相等，这个近似应保留在方法描述与实验限制里。</p>
<p>H20额外去噪GAN则从四步终点detach后在249重新加噪，多调用一次学生，再完整VAE解码接冻结SAM特征判别头。训练中的额外调用获得GAN梯度，正式评测仍是原四步。直接末步GAN是另一组，其正式训练止于G33，没有最终权重，不能用额外去噪组的25.6695 Patch FID替它背书。</p>
<p>所有GAN真实分支读取缓存真实latent后VAE重建图像，这个真实分布是VAE重建分布。COCO评测的真实参考来自原始图像。方法可以这样定义，但“更接近真实自然纹理”比当前证据更强。</p>
<p>A×10加局部GAN，FID 24.6412→24.8206，Patch 35.2295→33.9192，耗时6.82→11.15h（约+63.6%）。H20额外去噪GAN相对C＋1%，FID近乎持平，Patch明显下降，但硬件和计算量不同。旧四点末步纯GAN的Patch与纯DMD几乎相同。当前证据支持继续研究GAN局部分布收益，尚不支持GAN完全替代末步DM。</p>''')
section('对已有解释提出反证', table(['原解释或假设','反方核查','目前可以说到哪里'],[
['B差是因为fake跟不上高噪声','B绝对fake loss可能因高噪声任务更容易、时间权重不同而更低；没有相同t和相同学生分布下的充分拟合诊断','B在本配方指标变差成立，原因尚未定位'],
['去掉中段DM说明中段DM没有价值','删除loss同时降低共享梯度总量并改变PCM/DM相对权重，且中段fake仍保留','只支持“这个预算下学生仅末步DM的组合表现较好”'],
['PCM×10解释旧clean＋GAN的优势','A1→A10只改善FID，Patch和CLIP下降；旧新还差fake预算、学习率、Adam、裁剪阈值、目标与GAN','PCM是重要变量，无法解释全部收益'],
['中段clean目标应该更好','A中段clean相对A，FID−0.7946但Patch＋13.1394；它又保留了新A的加噪支持，未复原整套旧方法','中段目标单独更换没有产生全面改善'],
['1%混合能稳定fake','C→C＋1%只有小幅FID/Patch变化，单训练种子；混合也会偏置fake且Adam矩不重置','当前可视作候选稳态手段，尚未证明长期跟踪因果收益'],
['更多DM更有效率','全选DM234训练时间多30.6%，FID只下降0.2713、Patch下降1.0732，且fake多一次','有质量/计算量权衡，不能直接宣称训练效率提高'],
['FID改善意味着语义纹理一起改善','仅末步学生DM比C＋1% FID下降3.5277、Patch下降4.9430，CLIP1024却下降0.1356','必须联合指标与同提示词图像判断'],
['图像转向曲线已经确定学习阶段','512条教师轨迹单位距离归一化后总体前高后低，未得到独特稳定中段分界；这是教师轨迹统计','可指导采样假设，不能替代学生误差/监督需求诊断']]))
section('评测可靠性、显著性与数据', '''<p>COCO评测复用了固定10,000条提示词、按原始行索引构造的种子和真实特征。整图FID先中心方形裁剪、Lanczos缩放512，再进入clean Inception299；Patch路径是原生中心299。代码检查10000个特征索引连续、无重复，协方差与分数有限。BF16学生采样、FP32 VAE解码和uint8转换保持同协议。</p>
<p>自训四步模型用保存的schedule和CFG1，SDXL50参照用CFG6。训练教师用CFG8，故“原SDXL50参照”不等于精确复现训练教师的CFG8输出。这不妨碍同协议自训组比较，但不能用公开参照差值解释某项蒸馏损失独自造成的质量损失。</p>
<p>CLIP1024使用固定OpenAI ViT-B/32版本，逐条提示词和种子已做配对核查。A×10减A为−0.2215，已存提示词bootstrap 95%区间约[−0.3121,−0.1317]；仅末步学生DM减C＋1%为−0.1356，区间约[−0.2344,−0.0374]。这些区间量化提示词层面的差异，不涵盖训练随机种子不确定性。CLIP128、旧4/16张小样不能混入1024排名。</p>
<p>本次没有得到原始训练数据与COCO在图像内容、caption近重复层面的完整去重证据。只能确认来源和角色不同，不能保证零内容重叠。正式论文需要记录数据来源、许可、采样索引及重叠检测范围。20K缓存的存在也不能当作训练已扩大到20K。</p>
<p>为避免把后续状态写成已完成，三组3090补测仍保留“11月11日已启动，11月12日未能实时取回”的状态。A×10＋1%首128张日志显示约0.847张/秒、batch2、分配峰值10.84GiB；这只是启动证据，不是完整评分。</p>''')
section('后续实验顺序与停止条件', '''<p>第一优先级是取回现有三组完整COCO结果，检查每组10000项、有限FID/Patch、128固定样图和权重哈希，然后更新总表。对已完成组只复用结果。旧两组局部先验与A/B/C跨区间组分开解释。</p>
<p>第二优先级是完成一个真正的单变量采样比较。基线使用已有“C＋1%，PCM＋仅末步学生DM，中段fake保留”；新组只改末步加噪到低噪声30%。保留fake中段分布、两次更新、学习率、PCM系数、损失截断、混合起点、样本和种子。先做含完整更新、评估与最终保存的小规模GPU验证，再决定G160。原来fake仅末步两次的配方应标为另一项待选方案。</p>
<p>第三优先级才是fake低噪声降权消融。固定一个采样配方，比截断10与末步普通噪声MSE；中段的条件目标与权重保持原设置。对末步按固定时间桶记录未加权MSE、加权loss、裁剪前梯度和固定学生样本上的score方向变化。要检验后期跟不上，至少覆盖G80/120/160，不能仅靠G24。</p>
<p>PCM与DM相对强度的问题已有本地DM4×1/×2代码包，可作为独立后续控制。只做这一个系数变化，并在相同步数测PCM/DM参数梯度范数与余弦。结果若只是同G更好、同时间更差，应明确报告计算预算权衡。</p>
<p>GAN继续建立在质量最稳的已验证基线上。先一组末步联合，不同时加入新的先验、改变fake比例、PCM权重和模型架构。若仍要宣称纹理收益，至少同时看整图FID、中心/多位置Patch、CLIP与固定prompt盲看，关注过锐化、重复纹理和主体缺失。</p>
<p>在模型行为、基线与预算尚未稳定时，直接扩到20K或更大模型会增加排错成本。可以先用现有10K把目标和关键对照闭合。正式SOTA主张需要匹配论文协议的公开基准、充分训练预算以及训练重复验证，当前G160结果应定位为探索性实验。</p>''')
section('资源、归档与可恢复性', '''<p>11月9日已完成最后一份A×10＋1%权重的实验室备份，完整SHA-256为9958db19b7fb8ffdbc04adfacfa8aa7372b79c0f75e4d941f7b4586612f75be1。A800源文件随后删除。训练进度图6592张中，616张关键节点及prompt/seed记录存到实验室并逐文件校验，5976张密集节点删除；共释放9.24GiB，清理后A800数据盘曾可用33GiB。11月12日实时空闲空间未重新核实。</p>
<p>数据盘主要还包括20K缓存、原始源分片、评测模型与特征以及COCO固定样图。系统盘包含Python环境、基础模型和10K训练缓存。下一组启动需要“实际剩余空间＋最终权重及评测预算”核算。磁盘不足可以等待清理并再次检查GPU，不能无声退出后仍对用户报已排队可运行。</p>
<p>已完成权重按用户要求备份后删除云端副本是可行流程。源码、配置、数据索引、必要依赖版本、评测manifest、分数和少量样图应与权重归档关联。研究目录不宜继续积累大量每两步8张图；每40G加首尾和少量问题诊断点通常足够。用于计算指标的npz与展示PNG分别管理。</p>
<p>已有“只保留最终可测试权重”的决定意味着中断后从头重跑成本仍存在。这是用户选择的存储策略，不能在报告中误写成代码已经支持该权重精确续训。需要续训的少数组才保存fake、Adam和RNG完整状态。</p>''')
section('本次验证、修正和交付索引', '''<p>已完成21项核心CPU测试、3项DM权重消融测试，以及低噪声权重代数、采样控制变量和队列验收正反例复现。初次测试受Windows临时目录权限影响，改用项目测试目录重跑后全部通过。固定后验原始KL方向有限差分、fake条件目标、局部/分块梯度一致、SAM/VAE输入梯度、阶段配对、混合embedding保护均在覆盖范围内。</p>
<p>唯一实施的本次源码修正为本地tmp/run_terminal_low30.py完成验收：移除对不存在的逐G混合日志的要求，改用基线已有的verify_tracking_probes。修正前版本保存在evidence/run_terminal_low30_before.py。未改变训练损失或超参，未触发新训练。远端部署因SSH超时未执行。</p>
<p>对近期已核查的历史结果，报告提供逐项来源JSON与关键文件哈希。旧方法一部分耗时和H20分数仍来自已归档汇总/实验记录，报告没有声称本次逐份重算了全部FID。文献名称在此用于识别已有实现背景；本次没有重新下载或验证所有论文最新版，方法是否忠实原论文应另按官方版本逐式对齐。</p>'''+table(['交付文件','内容'],[
['report.html / 项目审计报告.txt','完整阅读版及纯文本版'],
['experiment_results.json','17组自训＋2公开参照的结果、CLIP样本量、时间与来源'],
['source_inventory.json','20个关键源码文件SHA-256与大小，记录本地审计版本'],
['evidence/core_tests.txt / ablation_tests.txt','21＋3项CPU测试结果'],
['evidence/audit_reproductions.json','低噪声权重数值、单变量先验验证、旧验收失败与新验收反例'],
['quality_tradeoffs.png / fake_low_noise_weight.png','可单独使用的科学绘图'],
['../实验记录.txt','按时间追加的运行、配置变更、归档与本次审计记录']]))
css='''body{margin:0;background:#f2f4f6;color:#182633;font:16px/1.78 "Microsoft YaHei",Arial,sans-serif}main{max-width:1160px;margin:32px auto;padding:36px 48px;background:#fff}h1{font-size:30px;margin:0 0 12px}h2{font-size:23px;border-top:1px solid #d8e0e7;padding-top:24px;margin-top:40px;color:#12394c}p{margin:12px 0}.meta{color:#536776;font-size:14px}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{border-bottom:1px solid #dce3e8;text-align:left;padding:11px 12px;vertical-align:top}th{background:#eaf1f5;white-space:nowrap}tr:nth-child(even){background:#f7f9fa}pre{background:#f1f5f7;padding:18px;white-space:pre-wrap;line-height:1.65}img{width:100%;height:auto;margin:15px 0}nav{display:flex;gap:8px 18px;flex-wrap:wrap;margin:22px 0}a{color:#17638a}nav a{font-size:14px}@media print{body{background:white}main{margin:0;padding:0;max-width:none}nav{display:none}h2{break-after:avoid}tr,img{break-inside:avoid}}'''
body='<h1>SDXL分阶段蒸馏项目 · 全面审计报告</h1><p class="meta">2026年11月12日｜历史结果核对、训练与评测代码反方审查｜最新远程补测状态待连接核验</p><nav>'+''.join(f'<a href="#s{i}">{html.escape(t)}</a>' for i,(t,_) in enumerate(sections))+'</nav>'
body+=''.join(f'<section id="s{i}"><h2>{html.escape(t)}</h2>{s}</section>' for i,(t,s) in enumerate(sections))
body+='<p class="meta">关键源码定位：boundary.py 36–79；train.py 161–168、452–563、751–976、1010–1045；turning_sampling.py 52–86；fake_tracking.py 31–82；tmp/run_terminal_low30.py 46–59；evaluation_core.py 69–78、192–288。行号对应source_inventory.json中的本地版本。</p>'
(out/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SDXL项目全面审计</title><style>'+css+'</style><main>'+body+'</main></html>',encoding='utf-8')
class Text(HTMLParser):
 def __init__(self):super().__init__();self.parts=[]
 def handle_starttag(self,t,a):
  if t in ['p','h1','h2','tr','pre','section']:self.parts.append('\n')
 def handle_endtag(self,t):
  if t in ['td','th']:self.parts.append(' | ')
  if t in ['p','h1','h2','tr','pre']:self.parts.append('\n')
 def handle_data(self,s):self.parts.append(s)
p=Text();p.feed(body);text=re.sub(r'\n{3,}','\n\n',''.join(p.parts));(out/'项目审计报告.txt').write_text(text,encoding='utf-8')
print('REPORT_READY',len(text),'characters',len(sections),'sections')
