from pathlib import Path
import sys,json,hashlib,html,zipfile
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'.plot_runtime_pkgs'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=root/'output/sdxl-crossdistill/optimization_plan_20261022'
out.mkdir(parents=True,exist_ok=True)
history=[
 {'name':'Joint4 / 4 A100','run':'gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16','mean_cycle':118.93582163039287,'median_cycle':115.60482025146484,'G':26.67825587411785,'fake_D':18.345171291589928,'peak_GiB':29.925310134887695},
 {'name':'Joint34 / 4 A100','run':'gan_stage_joint34_4gpu/joint34_20261019T195324_4gpu_mb16','mean_cycle':176.56055426142018,'median_cycle':177.7521209716797,'G':41.93739571866051,'fake_D':26.71774514813427,'peak_GiB':29.924283981323242},
 {'name':'Joint4 / 2 A100','run':'gan_stage_parallel/joint4_20261017T232317_mb16','mean_cycle':284.9806139715787,'median_cycle':284.10081481933594,'G':67.42556321977413,'fake_D':42.07020752991572,'peak_GiB':32.33875322341919},
]
wrapping={'fine':{'nonempty_groups':103,'largest_BF16_MiB':93.7744140625},'block':{'nonempty_groups':33,'largest_BF16_MiB':669.1748046875}}
evidence={'date':'2026-10-22','historical_logs_live_read':history,'wrapping_meta_count':wrapping,
 'gpu_optimization_benchmark':'not run','CPU_checks':'30 unit tests and 2-rank Gloo/CPU DTensor checks passed',
 'warning':'Historical runs used different GPU load and configurations; they are not controlled scaling results.'}
(out/'evidence/optimization_evidence.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
fig,ax=plt.subplots(figsize=(10,4.6))
names=[x['name'] for x in history]
fake=[5*x['fake_D'] for x in history];g=[x['G'] for x in history]
ax.barh(names,fake,label='5 fake/D updates',color='#297b95')
ax.barh(names,g,left=fake,label='Student + PCM',color='#edae49')
for i,r in enumerate(history):ax.text(fake[i]+g[i]+3,i,f"{r['mean_cycle']:.1f}s total",va='center')
ax.set_xlim(0,340);ax.set_xlabel('Seconds per G update, historical mean after first 3 updates')
ax.set_title('Historical workload distribution — not an optimization speedup comparison')
ax.legend(loc='lower right');fig.tight_layout()
fig.savefig(out/'historical_time.png');fig.savefig(out/'historical_time.pdf');plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(9.6,4))
for ax,key,title in [(axs[0],'nonempty_groups','Nonempty FSDP groups'),(axs[1],'largest_BF16_MiB','Largest BF16 parameter group (MiB)')]:
 vals=[wrapping[k][key] for k in ('fine','block')]
 ax.bar(['Fine','Block'],vals,color=['#8497a5','#297b95']);ax.set_title(title)
 for i,v in enumerate(vals):ax.text(i,v*1.02,f'{v:.1f}',ha='center')
 ax.set_ylim(0,max(vals)*1.2)
fig.suptitle('Static SDXL architecture count — fewer groups, larger working set',y=1.02)
fig.tight_layout();fig.savefig(out/'fsdp_groups.png',bbox_inches='tight');fig.savefig(out/'fsdp_groups.pdf',bbox_inches='tight');plt.close(fig)
scores=json.loads((root/'output/sdxl-crossdistill/coco10k_g160_20261020/snapshot_20261021/results.json').read_text())['scores']
keys=['dmd_only','gan4','joint4','joint34','dmd2'];labels=['TM+DMD','Step4 GAN','Step4 joint','Step3+4 joint','DMD2 official']
fig,axs=plt.subplots(1,2,figsize=(11,4.1))
for ax,key,title in [(axs[0],'FID','Full-image FID ↓'),(axs[1],'Patch_FID','Center-crop Patch FID ↓')]:
 vals=[scores[k][key] for k in keys]
 ax.bar(range(5),vals,color=['#8497a5','#8497a5','#297b95','#8497a5','#ddc2a1'])
 ax.set_xticks(range(5),labels,rotation=18,ha='right');ax.set_title(title);ax.set_ylim(0,max(vals)*1.16)
 for i,v in enumerate(vals):ax.text(i,v+.3,f'{v:.2f}',ha='center',fontsize=9)
fig.suptitle('COCO-2014 10K: saved 2026-10-21 snapshot; official DMD2 has a different training budget')
fig.tight_layout();fig.savefig(out/'existing_metrics.png');fig.savefig(out/'existing_metrics.pdf');plt.close(fig)

def table(headers,rows):
 return '<div class="table"><table><thead><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def link(url,label):return f'<a href="{url}">{label}</a>'
src={
 'fsdp':link('https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html','PyTorch FSDP2'),
 'adam':link('https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adamw.py','PyTorch 2.5.1 AdamW 源码'),
 'dmd2':link('https://github.com/tianweiy/DMD2','DMD2 官方仓库'),
 'sdxl':link('https://github.com/tianweiy/DMD2/blob/main/experiments/sdxl/README.md','DMD2 SDXL 配方'),
 'cross':link('https://github.com/AlibabaResearch/SparkDiffusion','CrossDistill / SparkDiffusion'),
 'rcm':link('https://github.com/NVlabs/rcm','rCM 官方实现'),
 'cdm':link('https://github.com/byliutao/CDM','CDM 官方实现'),
 'geneval':link('https://github.com/djghosh13/geneval','GenEval 官方实现'),
 'kid':link('https://github.com/toshas/torch-fidelity','torch-fidelity 指标参考'),
}
body='''<header><div class="eyebrow">SDXL RESEARCH · 2026-10-22</div><h1>训练加速与后续实验方案</h1><p class="lead">先把计算花在正确的对照上：保持训练目标，优化执行，再检验末步 GAN 的真实收益。</p></header>
<aside><b>交付状态：</b>独立训练代码、实验配置、可配置 COCO 评测和技术报告已完成。CPU/Gloo 验证通过；完整 SDXL 的 GPU 短测、端到端提速和新实验效果尚未测得。本次没有启动长训练、重置显卡或迁移模型资产。</aside>
<nav><a href="#evidence">已有证据</a><a href="#logic">训练逻辑</a><a href="#speed">加速实现</a><a href="#matrix">实验矩阵</a><a href="#evaluation">评测</a><a href="#budget">预算</a><a href="#next">SD3 与新方法</a><a href="#delivery">交付与运行</a></nav>
<section id="evidence"><h2>1. 当前证据支持什么</h2><p>下面是此前已保存的 COCO-2014 10K 结果（2026-10-21 快照），不是优化代码的新训练结果。末步 joint 的 Patch FID 比 TM+DMD 低约 <b>8.60%</b>，全图 FID 只低约 <b>0.084</b>。纯 GAN 末步没有对应的 Patch FID 收益。现有证据值得继续检验联合损失，但不足以声称 GAN 已改善普遍视觉质量或达到 SOTA。</p><img src="existing_metrics.png" alt="已有五组FID和Patch FID柱状图">'''
body+=table(['方法','FID ↓','Patch FID ↓','解释'],[[labels[i],f"{scores[k]['FID']:.3f}",f"{scores[k]['Patch_FID']:.3f}",'外部质量参照，训练预算不同' if k=='dmd2' else '单次短训练；未做多训练种子确认'] for i,k in enumerate(keys)])
body+='''<p>历史 gan4 使用两卡，其余部分对照使用四卡；虽然批量与梯度累积已有数值检查，也不能把这些记录当作严格一致的最终消融。原始 SDXL 在上述快照中尚未全部评完，本报告不补造它的分数。</p>
<h3>真实耗时，而非理论算力换算</h3><img src="historical_time.png" alt="历史训练耗时分布"><p>本次重新读取历史 rank0 日志，去掉最初三个 G 更新。Joint4 四卡平均 118.94 秒/G，五次 fake/D 合计约 91.73 秒，占 77.1%；G 更新约 26.68 秒。Joint34 四卡平均 176.56 秒/G，旧双卡 Joint4 为 284.98 秒/G。这些时间受共享负载、组别与 optimizer offload 设置影响，不能据此宣称四卡必然比两卡快 2.4 倍。图中总耗时与分项和的少量差异来自同步、日志与跨卡最大值口径。</p>
<p>原四卡峰值约 29.93GiB，旧双卡约 32.34GiB，均为 PyTorch allocated，而非完整进程占用。新代码同时记录 allocated 与 reserved 的跨卡最大值。4090 48GB 的可用性仍应以启动时健康检查和实际短测为准，不能从剩余显存推断故障卡可用。</p></section>
<section id="logic"><h2>2. 明确网络、损失和梯度路径</h2>
<div class="flow">5 × [student 四步造样本（无梯度） → fake score + 判别器更新]<br>↓<br>1 × [选第 2/3/4 步之一，循环选择 → DMD，末步加 GAN] + 高噪声 PCM<br>↓<br>更新同一个 student UNet；每 2 次 G 做小样本诊断</div>
<p>全程是一个共享参数的 student UNet，而不是四个采样点各有一套模型。“GAN 作用于第 4 步”表示该步预测的干净图像接收 GAN 梯度；前缀 rollout 被 detach，GAN 不贯穿全部四步反传，但最后一次 UNet 调用涉及的共享参数都能更新。fake、teacher 的评分在 G 分支中无梯度；VAE 与 DINO 的参数冻结，但 G 分支必须保留其对输入的梯度。</p>'''
body+=table(['分支','计算/梯度','保持的超参数'],[
 ['fake score','对 detach 后四步 student 样本加噪，拟合噪声；只更新 fake','每 G 更新 5 次；LR 1e-6；t∈[0,999]'],
 ['DMD','g=(fake_x0−teacher_x0)/mean|student_x0−teacher_x0|；target=stopgrad(student_x0−g)，对 student 做平方 surrogate','归一化下限1e-5；teacher CFG8；student CFG1'],
 ['GAN','D 为 RGB + 冻结 DINO 特征头；G loss=−D(fake)，D 用 hinge；局部 latent 解码','权重0.005274229682981968；D LR1e-4；R1每16次D，系数0.05×16'],
 ['TM/PCM','真实 latent 加高噪声，teacher 两次10步跳转；student 的边界预测匹配 stopgrad 目标','PCM权重10；每次G都计算；并非只在每第三个G训练'],
 ['优化器与批量','student/fake为AdamW；原FSDP平均与微批累积归一化保留','总batch64；micro16；2卡累积2、4卡累积1；同数据/种子/初始化'],
])
body+='''<h3>必须纠正的混淆：推理采样点 ≠ DMD 再加噪时间</h3><p>早先准备的 <code>sdxl_uniform4_20261022</code> 同时把 DMD 上限改为749，这会混入额外因素。本次新目录把 <code>schedule</code>、<code>pcm_boundary</code>、<code>dmd_t_max</code> 独立配置。<b>主实验普通四点用999/749/499/249；PCM边界749；DMD仍用20–919。</b>“DMD也改到20–749”单列为消融。旧分段的PCM边界919，PCM采样940–999；新分段为770–999。两者都保留teacher两次10步子步。</p>
<p>所以主2×2检验的是“分段策略”与“GAN”的交互，不能把分段收益进一步归因于只移动了推理节点。保留旧PCM边界的 sampler-only 配置可作诊断，但它故意不与新首段终点对齐，不宜作为最终默认配方。</p>
<h3>G160 到底训练了多少</h3><p>160次学生更新 = 10,240个学生样本位置；fake/D各800次 = 51,200个样本位置；PCM160次。阶段2/3/4分别54/53/53次，<b>末步 GAN 实际只有53次学生更新，3,392个样本位置</b>，不是160次末步GAN训练。更新次数、重复数据曝光量与唯一图片数必须分别报告。</p></section>
<section id="speed"><h2>3. 已实现的加速与验证边界</h2>'''
body+=table(['改动','作用','当前证据/代价'],[
 ['判别器梯度分桶','默认16MiB桶替代逐参数all-reduce；仍在D optimizer前求全局均值','CPU更新一致；2-rank Gloo真通信测试通过。实际GPU提速待测。'],
 ['周期边界控制','每个完整G周期检查停止/计时存档；去掉微批中的重复GPU广播和CPU同步','停止仍在完整周期边界；时间存档最多延后一个周期。每40次存档不变。'],
 ['foreach AdamW开关','批量执行CPU fake optimizer张量运算；student也提供独立开关','CPU DTensor8步参数差0；小模型约3.2倍仅是optimizer微测。旧较大微测无稳定收益，因此默认关闭。'],
 ['粗粒度FSDP候选','使用已有block包装减少通信组；不改变网络参数或损失','静态103→33组，最大组94→669MiB；显存、CUDA数值和速度待测。'],
 ['合批teacher CFG候选','把uncond/cond合成一次teacher前向','已有CPU CFG检查；激活显存增加，需实际SDXL梯度/输出检查。'],
 ['只编译冻结teacher','torch.compile teacher.forward，不直接编译多优化器FSDP训练循环','实现为opt-in，未GPU执行；需计入首次编译时间和形状重编译。'],
 ['真实训练比例benchmark','9次G、前3次warmup、fake:G=5:1、轮换2/3/4，包含正常R1节奏','修正原1:1短测不能估正式吞吐的问题；无eval/save的时间需另补。'],
 ['有限窗口Profiler','rank0 CPU/CUDA trace，带分支标签、形状与内存','与无Profiler吞吐测量分开，避免把测量开销当优化退化。'],
])
body+='<img src="fsdp_groups.png" alt="FSDP分组数及最大组体积"><p>FSDP 的通信前聚合、反向 reduce-scatter 和参数分组机制参考 '+src['fsdp']+'；具体API已在当前 PyTorch 2.5.1 环境检查，没有假定最新文档API都可用。foreach保留原AdamW公式，参考 '+src['adam']+'。'+src['dmd2']+' 自身也记录了 SDXL FSDP 较慢的问题，因此不把更换框架当作自动提速保证。</p>'
body+='''<h3>短测顺序与接受条件</h3><ol><li>先原数值路径与bucket/cycle；再分别测block、foreach、合批CFG、teacher compile，每次只改一项。选出候选后再组合测试。</li><li>同机器、同卡数、同batch与5:1比例；至少A/B/B/A复测。9次G仅筛选，最终18次G覆盖更多阶段和R1；保留每阶段中位数、全周期均值和峰值显存。</li><li>预注册工程门槛：至少约10%稳定端到端训练提速，无OOM/NaN/NCCL错误，峰值进程占用后至少留4GiB余量；toy测试不能代替完整SDXL短测。</li><li>同输入下比较G/fake/D输出、loss与梯度；BF16路径先记录baseline重复误差，以此定容差，不要求不同kernel逐bit一致。拒绝梯度断路、更新次数变化或损失路由变化。</li><li>正式前额外跑4次真实G短测，包含第4步GAN、D16的R1、每2次评估和最终可恢复存档；用保存/加载/下一次更新验证恢复。正式从原始SDXL重新初始化，不接短测权重。</li></ol>
<p>不采用会混淆科学结论的“优化”：减少fake次数、降低训练分辨率、替换GAN、减少teacher CFG、复用已过期fake/teacher输出、无验证地向量化随机数、用量化/LoRA替代当前全参训练。暂停后的恢复要求同卡数、FSDP布局与关键配置；最后完整状态只保存一份，不恢复大量冗余节点。</p></section>
<section id="matrix"><h2>4. 后续实验：先做可证伪的最小矩阵</h2><p><b>核心假设：</b>末步局部GAN与DMD联合，在固定teacher、数据和更新预算下改善局部图像分布，同时不显著损害全图质量、语义一致性或多样性。若只有“更锐”但伪纹理、结构或多样性变差，就否定该版本，不继续扩大训练量。</p>'''
body+=table(['优先级','配置文件','改动与回答的问题'],[
 ['主矩阵 A','legacy_dmd.json','旧分段 + TM/DMD，无判别器；同新版执行路径重新跑参照'],
 ['主矩阵 B','legacy_joint4.json','旧分段 + 末步joint，B−A是旧分段GAN效应'],
 ['主矩阵 C','uniform_dmd.json','普通分段 + TM/DMD，C−A是分段策略效应'],
 ['主矩阵 D','uniform_joint4.json','普通分段 + 末步joint，D−C是新分段GAN效应；(D−C)−(B−A)是交互'],
 ['有正信号再做','uniform_gan4 / uniform_joint34','末步纯GAN能否替代DMD？扩到第三步是否有额外收益？'],
 ['权重/边界消融','uniform_joint4_gan_half/double / uniform_aligned_dmd749','GAN系数0.5×/2×；单独检查DMD再加噪支持域'],
 ['诊断，不是默认','sampler_only_joint4','只改推理点、保留旧PCM与DMD支持域，检查“分段移动”混杂'],
])
body+='''<p>所有主配置都使用相同原始SDXL初始化、10K数据、seed20261017、batch64、LR1e-6、fake:G5:1、1024分辨率与相同runtime preset。第一次每组G160，保存40/80/120/160。旧单种子历史数值只作背景，不代替新runtime下的对照。</p>
<p>第一阶段只判断是否值得继续。若joint没有可见收益，优先检查GAN/DMD梯度比例、D真假评分与训练/留出差距、完整解码与局部解码差异，不立即加数据或提高GAN权重。GAN系数最早仅两个样本标定，正式训练D会快速变化；不能认为0.00527始终保持20%梯度比例。</p>
<p>第二阶段只保留最好的joint与其无GAN对照，做长一点的固定预算确认，建议至少3个成对训练种子分批完成。固定种子、初始化、样本顺序，报告成对差值与置信区间。10K张生成图片不能替代独立训练种子；对一个模型的样本bootstrap也不能证明训练方差。</p>
<h3>预先写下停止条件</h3><p>工程失败立即停止该候选：非有限梯度/图像、通信错误、OOM、结构异常反复出现。方法筛选门槛建议预注册为：Patch FID至少相对下降5%，全图FID恶化不超过1.0点，盲评纹理收益不依赖伪锐化。进入确认阶段后再要求成对种子结果一致；GenEval下降超过0.02或语义/多样性明显变差时，不以Patch FID单项收益宣称成功。上述阈值是本项目筛选标准，不是社区统一SOTA标准。</p></section>
<section id="evaluation"><h2>5. 数据和评测协议</h2><p>实际训练数据是DMD2提供的LAION VAE latent分片000、001中准备的10,000张训练样本，另有32张留出；不是COCO训练集。本次精确文本核验：训练内规范化提示词10,000个唯一；训练/32留出重叠0；训练/COCO10K提示词重叠0。见 <a href="evidence/data_audit.json">data_audit.json</a>。这不能排除图像近重复、语义相似或teacher预训练重叠。</p>
<p>先保持10K，回答方法问题。32张留出只够监控，下一阶段从未用训练来源中构建固定512–1,024条开发样本，做规范化caption、原始图片/latent哈希与近重复检查。只有确认过拟合且GAN有稳定收益后，做10K→50K的同更新数、同曝光预算对照；不把一次下载500K作为前置条件。数据扩展需要重新生成manifest和文本编码，当前代码没有伪造已准备好的50K目录。</p>'''
body+=table(['评测层级','内容','用途与限制'],[
 ['每2次G诊断','固定4训练+4留出图；D总分、DINO/RGB分支真假分；训练loss/梯度/R1','监控崩溃与D失衡，不能称benchmark'],
 ['开发筛选','独立开发prompt；KID、固定多seed整图/局部盲评；可选COCO1K诊断','KID保留raw单位；subset_std不是训练种子CI。若用COCO调参就承认它是开发集。'],
 ['最终质量','固定COCO-2014 10K；原FID与native center299 Patch FID；保存所有Inception特征','自定义模型使用各自checkpoint.schedule；同分辨率、VAE精度与seed策略'],
 ['语义与组合','GenEval作为新方法外部检查；CLIP仅辅助','不因追求纹理牺牲对象数量、颜色、位置；GenEval调用仍使用官方工具'],
 ['纹理与多样性','至少100条固定提示词，4噪声seed；随机盲排整图+相同坐标裁剪；检查重复纹理、伪轮廓、跨seed差异','“更锐”不等于更真；单中心Patch FID不覆盖全图纹理，也不直接度量文字语义'],
])
body+='<p>GenEval的定位参考 '+src['geneval']+'；KID等通用指标可对照 '+src['kid']+'。本次已经实现COCO可配置评测和保存特征上的KID，尚未把GenEval/多样性评测封装成新的训练框架。</p>'
body+='''<h3>保持评测可比性</h3><p>全图FID保持旧DMD2口径：中心正方形→Lanczos512→clean Inception299。Patch FID保持原生中心299裁剪（不足299时沿用原规则），不是训练256px局部patch的直接误差。自定义模型用确定性VP/DDIM跳转；官方DMD2用原研究x0再加噪采样器。即使四个时间点相同，两者也不是同一个采样器。</p>
<p>GAN训练对48×48 latent局部解码，再去掉64px边缘得到256×256图像；VAE空间归一化使它与整图解码后裁剪并不等价。先在同一latent/坐标上比较二者的纹理与误差分布。若差异驱动判别器，应把“整图解码后裁剪”单列实验（显存会变），不能偷偷作为性能优化替换。</p></section>
<section id="budget"><h2>6. 48–72小时预算，而不是凭卡数猜天数</h2><p>以下以本次读到的历史均值119秒/G（四卡）与285秒/G（双卡）举例，仅用于规划，<b>不是4090实测预测</b>。正式以新机器、真实5:1、无Profiler短测替换。teacher/学生都可能受并行通信和CPU offload影响，不能按4090与A100的算力比例换算。</p>'''
body+=table(['计划','119秒/G情景','285秒/G情景','包含什么'],[
 ['单组G160','5.29小时训练','12.67小时训练','另加初始化、每2次诊断、权重及最终完整存档'],
 ['2×2四组G160','21.16小时训练','50.67小时训练','再为统一评测/IO预留约8–12小时；先做一次真实eval吞吐测量'],
 ['72小时两组确认；预留12小时评测/IO','每组最多906次G','每组最多378次G','按整三阶段周期向下取整；并非保证收敛'],
])
body+='''<p>计算式：<code>每组G上限 = floor_to_3((总小时−评测IO预留)×3600 / (组数×实测秒/G))</code>。若完整COCO10K评测超出预留，则降低更新预算或拆开发与最终评测，不把评测成本隐藏到预算外。<code>budget.py</code> 已实现这个计算。三种子确认分多个窗口进行，不承诺把所有矩阵、长训、三种子和全部benchmark一次塞进两天。</p>
<p>存储保留每40次学生快照与最后可续训状态。长训超过G160时需另定保留数量；旧成果不自动删除。评测仅保留固定128张PNG/模型与全部FP32特征，避免再次累计大量整图和完整optimizer节点。</p></section>
<section id="next"><h2>7. 现代基线与 SD3-Medium 迁移</h2>'''
body+=table(['方法','本项目角色','可比性要求'],[
 ['原始SDXL、官方DMD2','当前SDXL质量上限/外部蒸馏参照','统一prompt、分辨率、指标；官方权重预算不同，不能当同成本因果消融'],
 ['CrossDistill思想迁移','当前TM+分布匹配分段研究来源','原实现面向Wan视频；不能把这里的SDXL混合实现称原论文完整复现'],
 ['rCM（2025/2026）','现代一致性+score正则机制与工程参照','公开主线面向视频/Wan；没有核验到同teacher SDXL权重时，不把视频分数排进图像FID表'],
 ['CDM（2026）','SD3-Medium阶段优先纳入的现代图像蒸馏参照','已有官方SD3-Medium推理/训练入口；固定权重revision、清理训练/测试prompt重叠后再比'],
])
body+='<p>来源：'+src['sdxl']+'、'+src['cross']+'、'+src['rcm']+'、'+src['cdm']+'。DMD2文档的4-step条目报告19k iterations/57h，示例为64 GPU；其iterations口径需结合generator更新比例解析，不能把它与本项目G160直接相除或按GPU数量线性缩放。</p>'
body+='''<p>SD3阶段先复现官方推理与一批loss/梯度，再做两卡/四卡小测。需要替换UNet→MMDiT、SDXL双文本编码→SD3条件、VP噪声/ε预测→正确的flow/velocity参数化、scheduler实际sigma与time映射、VAE scaling/shift。不能直接把999/749/499/249塞入SD3作为等价噪声，也不能复用SDXL teacher转向曲线。优先预编码文本并复用当前日志/评测设计，FSDP按transformer block重新确定；模型/optimizer/激活预算重新测量。</p>
<p>先完成SDXL中GAN相对DMD的可重复收益，再迁移。CDM训练数据与测试提示词的重叠风险沿用此前审查结论，迁移时用新split重新核验。本次交付没有实现SD3训练器、rCM图像复现或宣布任何2026榜单SOTA。</p></section>
<section id="delivery"><h2>8. 代码、检查结果和执行顺序</h2>'''
body+=table(['文件','用途'],[
 ['sdxl_research/train.py / runtime.py','独立优化训练器、D梯度分桶、控制通信、Profiler、分离时间变量'],
 ['configs/*.json','主2×2、GAN范围/权重/时间域消融；性能候选单变量配置'],
 ['launch.py / benchmark_suite.py / summarize.py','显式单次启动与健康检查；默认只打印命令；真实比例性能汇总'],
 ['evaluate.py / evaluation_core.py / feature_metrics.py','checkpoint采样点驱动的COCO10K、原FID/Patch FID、离线KID'],
 ['audit_data.py / budget.py','精确prompt重叠核查；按实测速度分配预算'],
 ['test_runtime.py / test_experiments.py / test_distributed_cpu.py','输入/梯度/时间点/指标/配置与CPU分布式验证'],
 ['README.txt / environment.json','完整命令、目录布局、环境版本、尚未验证的边界'],
])
body+='''<p><a href="evidence/cpu_tests.log">30项CPU测试日志</a> · <a href="evidence/distributed_cpu.log">两进程Gloo/DTensor结果</a> · <a href="evidence/environment.json">已测环境版本</a> · <a href="evidence/optimization_evidence.json">历史性能与分组统计</a> · <a href="sdxl_research_20261022.zip">代码与配置压缩包</a></p>
<p>已验证：采样范围与原始随机流；global batch/梯度累积；裁剪/R1/冻结DINO梯度；D梯度分桶与Adam更新；CPU DTensor foreach；评测确实使用checkpoint新四点；KID与显式U统计公式一致。<b>未验证：</b>新完整训练器CUDA执行、block-FSDP/compile的真实SDXL数值、目标4090端到端吞吐、目标环境安装与新实验效果。旧FSDP可恢复测试脚本保留，需在实际选定GPU上再次执行。</p>
<pre>ROOT=/data1/shenglin.zeng/sdxl_crossdistill
# 仅打印命令；先准备匹配环境与资产，并确认GPU归属
python sdxl_research/launch.py --root "$ROOT" --gpus 2,7 \
  --config sdxl_research/configs/uniform_joint4.json --short
# 真正执行时显式加 --execute；短测通过后从原始权重另启正式run。
# 性能矩阵也默认dry-run，避免自动占用共享GPU。
python sdxl_research/benchmark_suite.py --root "$ROOT" --gpus 2,7 \
  --candidates legacy,bucket_cycle,block</pre>
<p>2、7只是最近确认过空闲的候选，不是长期预约。4、5有反复reset状态，不使用；启动器会检查具体卡健康、现有进程与余量，但不能替代管理员的资源分配。此次只有原A100环境中的CPU验证，不存在已经在4090训练的进程。</p></section>
<footer>科学结论以同协议新实验为准。报告日期2026-10-22；指标为明确标注的历史快照，优化收益不以静态计数或toy微测冒充。</footer>'''
style='''*{box-sizing:border-box}body{margin:0;background:#f5f7fa;color:#1e2d3b;font:16px/1.8 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1120px;margin:auto;padding:40px 28px}header{padding:20px 0}h1{font-size:38px;line-height:1.3;color:#143b52}h2{font-size:25px;line-height:1.5;margin-top:0}h3{font-size:19px;margin:24px 0 10px}.eyebrow{letter-spacing:2px;color:#397b90;font-weight:bold}.lead{font-size:19px;color:#526673}aside{background:#e4f0f4;border-left:5px solid #297b95;padding:20px}nav{display:flex;gap:14px;flex-wrap:wrap;padding:20px 0}a{color:#176984}section{background:white;border:1px solid #dde4e9;border-radius:10px;margin:20px 0;padding:28px}p{margin:12px 0}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border-bottom:1px solid #dbe3e9;text-align:left;padding:11px;vertical-align:top}th{background:#eef4f7;color:#23485b}td:first-child{min-width:130px}img{width:100%;height:auto;margin:18px 0}code,pre{font-family:Consolas,monospace}code{background:#eef3f6;padding:2px 4px}pre{background:#182e40;color:#e1eef5;padding:20px;white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}.flow{background:#edf5f8;text-align:center;padding:24px;line-height:2.2;border-radius:8px}footer{color:#637888;font-size:14px;padding:20px}@media(max-width:700px){main{padding:20px 10px}section{padding:18px}h1{font-size:28px}}'''
(out/'report.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>SDXL训练优化与实验方案</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>',encoding='utf-8')
# Preserve exact artifact source identity. CPU tests are not a CUDA performance claim.
project=root/'sdxl_research'
env=json.loads((out/'evidence/environment.json').read_text())
(project/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8')
hashes={p.relative_to(project).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in project.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='source_manifest.json'}
(project/'source_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
with zipfile.ZipFile(out/'sdxl_research_20261022.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in project.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,'sdxl_research/'+p.relative_to(project).as_posix())
 z.write(root/'sdxl_parallel/batching.py','sdxl_parallel/batching.py')
print(out/'report.html')
