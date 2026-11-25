import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';
const work='C:/Users/sherl/Documents/ChatGPT/推理加速';
const build=work+'/tmp/weekly_1014_v2';
const skill='C:/Users/sherl/.codex/plugins/cache/openai-primary-runtime/presentations/26.1011.11113/skills/presentations';
const final=work+'/output/pptx/周组会_少步扩散蒸馏论文梳理_20261014_v2.pptx';
const {finalizePresentation}=await import(pathToFileURL(skill+'/container_tools/artifact_tool_utils.mjs').href);
const C={bg:'#F5F8FC',ink:'#172436',muted:'#5C6F84',cyan:'#159DB2',blue:'#3778D8',violet:'#8059C8',orange:'#D98525',green:'#23986F',red:'#D95863',line:'#CBD7E4'};
const FONT='Microsoft YaHei';
const p=Presentation.create({slideSize:{width:1280,height:720}});
await fs.mkdir(build+'/rendered',{recursive:true});await fs.mkdir(path.dirname(final),{recursive:true});
function txt(s,v,x,y,w,h,size=24,color=C.ink,bold=false,align='left'){
 const t=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 t.text=v;t.text.style={typeface:FONT,fontSize:size,color,bold,alignment:align,verticalAlignment:'top',autoFit:'none'};return t;
}
function rect(s,x,y,w,h,fill){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}})}
function slide(title,sub){const s=p.slides.add();s.background.fill=C.bg;rect(s,0,0,12,720,C.cyan);txt(s,title,62,35,1100,56,38,C.ink,true);txt(s,sub,65,99,1120,31,18,C.muted);txt(s,String(p.slides.items.length).padStart(2,'0'),1170,43,52,30,17,C.muted,false,'right');return s}
function notes(s,t,src=[]){s.speakerNotes.text=t+'\n\n参考资料\n'+src.join('\n')}
function method(o){const s=slide(o.title,o.sub);const color=o.color||C.blue;
 txt(s,'核心问题',65,157,500,32,21,color,true);txt(s,o.problem,65,207,495,158,25,C.ink,true);
 rect(s,619,155,1.5,455,C.line);txt(s,'训练时的改动',674,157,520,32,21,color,true);txt(s,o.method,674,207,520,184,23);
 txt(s,o.keyLabel||'关键机制',65,415,500,32,21,color,true);txt(s,o.key,65,462,495,124,24,color,true);
 txt(s,'本页结论',674,415,520,32,21,color,true);txt(s,o.result,674,462,520,124,24,C.ink,true);
 txt(s,o.footer||'',65,650,1130,28,16,C.muted);notes(s,o.notes,o.sources);return s;
}
function table(s,values,widths,top,height,size=22){const t=s.tables.add({rows:values.length,columns:values[0].length,left:64,top,width:1152,height,columnWidths:widths,values});
 for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?'#EAF2FA':(r%2?'#FFFFFF':'#F5F8FC');cell.text.style={typeface:FONT,fontSize:size,color:r===0?C.cyan:C.ink,bold:r===0,verticalAlignment:'middle',autoFit:'none'};}
 t.borders.assign({fill:C.line,width:.6,style:'solid'});return t;
}
const src={scm:'https://arxiv.org/abs/2410.11081',rcm:'https://arxiv.org/abs/2510.08431',rt:'https://arxiv.org/abs/2605.26108',sg:'https://arxiv.org/abs/2605.30116',co:'https://arxiv.org/abs/2606.21982',dy:'https://arxiv.org/abs/2609.31349',flash:'https://arxiv.org/abs/2511.20549'};
{
 const s=p.slides.add();s.background.fill=C.bg;rect(s,0,0,18,720,C.cyan);
 txt(s,'扩散模型推理加速周报',80,139,1110,86,54,C.ink,true);
 txt(s,'少步蒸馏论文梳理',84,239,1080,52,32,C.cyan,true);
 txt(s,'RTDMD（含 AC-DMD）、SGMD、CoDMD、DyMD\nrCM（含 sCM）、Flash-DMD',84,347,1100,100,26);
 txt(s,'组会汇报  2026.10.14',84,604,700,34,20,C.muted);
 notes(s,'本次汇报围绕少步蒸馏的训练机制，围绕六项主要工作展开，sCM 作为 rCM 的方法基础介绍。默认听众了解 DMD 的 teacher、student 和 fake score 三个角色。汇报重点是每篇解决什么问题、更新哪些网络，以及方法之间的区别。');
}
{
 const s=slide('本次论文的改进位置','以 DMD 的学生更新和 fake 跟踪为参照');
 table(s,[['改进位置','对应工作','主要问题'],['一致性训练','rCM（含 sCM）','怎样稳定学习终点映射，并补足生成质量'],['fake 跟踪与更新','AC-DMD、SGMD、DyMD','学生分布持续变化，fake 如何跟得上'],['样本与帧间关系','CoDMD','怎样约束视频布局和运动的关联结构'],['奖励与训练分工','RTDMD、Flash-DMD','怎样兼顾少步生成质量与奖励提升']],[235,355,562],176,360,23);
 txt(s,'共同参照',66,575,220,32,22,C.cyan,true);
 txt(s,'DMD 用 teacher / fake 的 score 差训练学生，fake 单独学习学生分布。',66,621,1135,42,24,C.ink,true);
 notes(s,'分类只用于讲解，同一论文可能涉及多个位置。AC-DMD 是 RTDMD 的第一阶段方法。sCM 并入 rCM 页，用作连续时间一致性训练的基础背景。无需逐行朗读表格。',Object.values(src));
}
method({title:'rCM：sCM 与分布匹配联合训练',sub:'Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency',color:C.violet,
 problem:'单独使用连续时间一致性训练，\n大规模图像和视频生成仍可能\n出现细节失真。',
 method:'一致性分支（sCM）：真实数据加噪，\n用 JVP 沿教师方向构造一致性监督。\n分布分支（DMD）：学生生成样本，\n由 teacher / fake 差异提供监督。\n两份梯度累积，更新同一个学生。',
 key:'L_student = L_sCM + λ L_DMD\nsCM 学终点一致性\nDMD 学生成分布',
 result:'sCM 作为 rCM 的一致性训练基础。\nDMD 分支补充生成分布监督。\nfake 单独更新，推理只用学生。',
 footer:'sCM 的 TrigFlow 与稳定训练设计作为基础，JVP 只在训练时使用',
 notes:'先在 rCM 内介绍 sCM 基础：同一条教师轨迹上的不同位置应预测同一终点。真实数据加噪，教师提供局部生成方向，JVP 计算学生输出沿位置和时间共同变化的方向导数，并据此构造连续时间一致性监督。sCM 通过 TrigFlow、目标归一化、自适应权重和预热稳定训练。rCM 保留这条一致性训练流程，再新增 DMD 分支：学生自行生成，teacher 与 fake 的差异提供分布监督。联合阶段两份损失加权求和，或分别反传后累积梯度，再做一次学生更新。交替的是学生更新与 fake 更新。论文随机选择一至四步 rollout，DMD 主要通过最后一次学生预测反传，前面的步骤停止梯度。训练初期有 warmup。sCM 在历史上是独立论文，本次仅作为 rCM 基础介绍，不单列汇报。',sources:[src.rcm,src.scm]});
method({title:'AC-DMD：给 fake 加跨时间一致性',sub:'RTDMD 第一阶段：Ambient-Consistent Distribution Matching Distillation',color:C.blue,
 problem:'学生不断更新，fake 需要追踪\n变化的分布。\n有限次更新下，不同时间的预测\n容易不协调。',
 method:'在生成过程的子区间内做分布匹配。\nfake 同时做去噪回归与一致性训练。\n当前预测应与反向采样后预测的\n条件平均相容。',
 key:'L_fake = L_denoise + γ L_consistency',
 result:'把不同时间的 fake 预测联系起来，\n改善有限预算下的跟踪。\n一致性约束施加在 fake 上。',footer:'AC-DMD 是 RTDMD 中的方法，和 sCM 约束学生终点的作用位置不同',
 notes:'子区间起点本身可能已经带噪，需要使用对应的 ambient 分布匹配目标。跨时间关系是期望一致性，不要求每条随机路径的预测逐点相等。论文通过反向采样估计条件期望，并使用双重独立采样构造一致性目标估计。这是 fake 估计器上的结构约束，与 sCM 对生成映射的一致性监督不同。',sources:[src.rt]});
method({title:'RTDMD：蒸馏后继续奖励优化',sub:'Reinforcing Few-step Generators via Reward-Tilted Distribution Matching',color:C.blue,
 problem:'少步学生会生成图像后，如何提高偏好和组合能力，同时保留教师的生成分布约束？',
 method:'第一阶段用 AC-DMD 冷启动。\n第二阶段继续分布匹配和奖励优化。\n中间随机转移使用 SubGRPO。\n最后确定性步骤用可微奖励反传。',
 key:'L_total = L_AC-DMD − β E[r]\n此处为概念表达，奖励项有两条梯度路径',
 result:'前面步骤学习高奖励转移。\n最后一步直接改善输出。\n分布匹配持续约束奖励训练。',footer:'SubGRPO 只选择部分步骤做策略更新，共享其他步骤的噪声以降低方差',
 notes:'理论目标来自奖励倾斜的教师分布 p_tilt 正比于 p_teacher 乘 exp(beta r)。KL 目标可分为分布匹配与期望奖励。四步示例中前三步是随机转移，最后一步噪声系数为零。SubGRPO 在组内共享未选步骤的噪声，使用最终奖励计算优势，并更新选中的转移。可微奖励只通过最后一次生成反传，无需展开全部步骤。训练 RM 分数提高不自动保证独立人类偏好提升。',sources:[src.rt]});
method({title:'SGMD：调整学生与 fake 的耦合更新',sub:'Score Gradient Matching Distillation for Few-Step Video Diffusion Distillation',color:C.orange,
 problem:'频繁更新 fake 的成本很高。\n减少更新时，fake 跟踪滞后，\n又会干扰学生学习。\n视频运动可能随之减弱。',
 method:'学生使用 teacher 停止梯度的 Fisher 匹配，再加入负残差修正 NR。\nfake 使用正残差回归 RC。\n两种更新配合，降低对频繁 fake 更新的依赖。',
 key:'学生：Fisher + λ NR\nfake：λ RC\nNR / RC 使用相反符号的残差项',
 result:'改动学生与 fake 的整个更新系统。\n学生梯度经过冻结 fake 的输入路径。',footer:'论文报告约 3 倍训练加速，来自减少 fake 更新次数',
 notes:'定义 Δ=x_fake−stopgrad(x_teacher)，r=stopgrad(x0)−x_fake。学生损失为二分之一 c(t)||Δ||²−二分之一 λ||r||²，fake 损失为二分之一 λ||r||²。学生更新时冻结 fake 参数，但保留 fake 对输入的梯度；fake 更新时切断学生梯度。RC 仍然是学生样本上的去噪回归，不是直接用 teacher 标签更新 fake 参数。残差混合了拟合误差和后验不确定性，因此不能将 NR 无条件理解为完全不改变正确平衡点的误差消除器。加速是特定训练设置的迭代时间收益，不是同为四步模型的推理速度提升。',sources:[src.sg]});
method({title:'CoDMD：增加样本和帧间关系约束',sub:'Copula-aware Distribution Matching Distillation for Fast Video Generation',color:C.green,
 problem:'少步视频蒸馏可能出现布局不稳、过饱和和运动破碎。\n逐样本的 DMD 目标缺少显式关系监督。',
 method:'复用 teacher 与 fake 已计算的 score 估计。\n分别构造跨样本、跨帧的关系矩阵。\n用额外的关系分布匹配目标，补充普通 DMD 训练。',
 key:'L_student = L_DMD + λ L_relation\n关系来自已有的 score 估计',
 result:'增加对关联结构的显式监督。\n无需新增网络、数据集或采样轨迹。',footer:'原始视频模型仍能表达帧间依赖，CoDMD 补充的是训练目标中的显式约束',
 notes:'Copula 可理解为描述联合分布依赖结构的工具。无需在组会上展开 copula 理论。重点是它不只比较每个样本的 score 值，还比较样本之间与视频帧之间的相对关系。源论文在 Wan2.1 1.3B 与 14B 上做四步蒸馏。低开销指复用已有估计且无需新网络，关系矩阵计算本身仍有成本。',sources:[src.co]});
method({title:'DyMD：按交互表现调整监督和预算',sub:'Preserving Interaction Dynamics through Distribution Matching Distillation in Few-Step Video World Models',color:C.red,
 problem:'机器人视频蒸馏后，画面质量尚可，\n但机器人与物体的交互运动减弱。\n强运动样本也更难让 fake 拟合。',
 method:'教师侧：按 rollout 的交互表现，\n自适应选择重加噪时间步。\nfake 侧：用带噪声条件的预测器，\n根据 latent 动态估计拟合难度，\n再提高困难样本的训练权重。',
 key:'教师监督：调整重加噪分布\nfake 拟合：预测难度并加权',
 result:'同时处理运动指导不足与 fake 跟踪困难。\n辅助预测器只在训练时使用。',footer:'应用重点是具身交互视频世界模型，结果需结合交互指标理解',
 notes:'弱重加噪时教师后验容易集中在原本运动不足的 rollout 附近，难以恢复运动。时间亲和度条件化的采样器混合基础时间步分布与教师先验，以兼顾运动恢复和外观细化。Dynamics-guided fake-score tracking 使用噪声条件化难度预测器，从 latent 动态估计相对噪声水平的拟合难度，再提高预测困难 rollout 的 critic 损失权重。原论文将 14B teacher 蒸馏为四步 1.3B student。其难度加权不等于证明任何干净样本依赖权重都保持原始 score 最优解。',sources:[src.dy]});
method({title:'Flash-DMD：按噪声阶段安排训练任务',sub:'Towards High-Fidelity Few-Step Image Generation with Efficient Distillation and Joint Reinforcement Learning',color:C.cyan,
 problem:'DMD 与 GAN 直接叠加，分工不足。\n少步学生继续做奖励优化时，\n也容易产生失真和奖励投机。',
 method:'高噪声阶段以 DM 学全局结构。\n低噪声以 Pixel-GAN 强化真实感。\n后续加入少步模型的奖励优化，\n同时保留蒸馏训练。',
 key:'高噪声：分布匹配\n低噪声：Pixel-GAN\n后续：蒸馏与 RL 联合优化',
 result:'先按时间步区分结构与细节任务。\n再用持续蒸馏稳定奖励优化。',footer:'与 RTDMD 的区别在于时间步分工和 Pixel-GAN，而非 AC-DMD 与 SubGRPO',
 notes:'Flash-DMD 的 early / later phase 描述训练阶段，高噪声 / 低噪声描述生成时间步，两种概念需要区分。第一阶段采用 timestep-aware 的 DM 与 Pixel-GAN 分工，并改善 score estimator，解耦 fake 去噪与判别职责。第二阶段继续蒸馏的同时进行面向少步模型的奖励优化。论文报告的训练成本比较依赖具体模型和实验设置，本汇报不将该比例推广到所有方法。',sources:[src.flash,'https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Flash-DMD_Towards_High-Fidelity_Few-Step_Image_Generation_with_Efficient_Distillation_and_CVPR_2026_paper.pdf']});
{
 const s=slide('方法对照','每篇论文主要改了什么，约束作用于哪里');
 table(s,[['工作','主要改动','作用位置'],['rCM（含 sCM）','sCM 一致性与 DMD 联合','同一个学生，两份监督'],['AC-DMD / RTDMD','fake 一致性，再联合奖励优化','fake 估计与学生的生成转移'],['SGMD','Fisher 匹配与 NR / RC 配对','学生与 fake 的交替更新'],['CoDMD','跨样本和跨帧关系匹配','学生的关系分布目标'],['DyMD','自适应重加噪与困难样本加权','教师指导位置与 fake 训练预算'],['Flash-DMD','时间步分工，蒸馏与 RL 联合','DM、GAN 与奖励训练']],[205,460,487],165,470,21);
 notes(s,'容易混淆的两组：sCM 与 AC-DMD 都谈一致性，但前者约束学生的生成映射，后者约束 fake 的预测。RTDMD 与 Flash-DMD 都结合奖励优化，但前者强调奖励倾斜分布目标、AC-DMD 和混合策略梯度，后者强调按噪声阶段分工 DM 与 Pixel-GAN。SGMD 与 DyMD 都关心跟踪成本，但前者改更新规则，后者改时间步和样本分配。',Object.values(src));
}
{
 const s=slide('参考论文','六项工作与 sCM 基础文献，原文链接保存在本页备注');
 const refs=[['sCM 基础','Simplifying, Stabilizing and Scaling Continuous-Time Consistency Models','2410.11081'],['rCM','Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency','2510.08431'],['RTDMD','Reinforcing Few-step Generators via Reward-Tilted Distribution Matching','2605.26108'],['SGMD','Score Gradient Matching Distillation for Few-Step Video Diffusion Distillation','2605.30116'],['CoDMD','Copula-aware Distribution Matching Distillation for Fast Video Generation','2606.21982'],['DyMD','Preserving Interaction Dynamics through Distribution Matching Distillation in Few-Step Video World Models','2609.31349'],['Flash-DMD','Towards High-Fidelity Few-Step Image Generation with Efficient Distillation and Joint Reinforcement Learning','2511.20549']];
 refs.forEach((r,i)=>{let y=163+i*68;txt(s,r[0],66,y,147,36,22,C.cyan,true);txt(s,r[1],223,y,817,54,20);txt(s,r[2],1060,y,155,35,18,C.muted,false,'right')});
 notes(s,'本汇报围绕方法机制展开，未对不同数据、骨干和评测条件下的指标做横向排名。参考文献链接如下。',Object.values(src));
}
await (await PresentationFile.exportPptx(p)).save(build+'/draft.pptx');
for(let i=0;i<p.slides.items.length;i++){
 const slide=p.slides.items[i];await fs.writeFile(`${build}/rendered/slide-${String(i+1).padStart(2,'0')}.png`,new Uint8Array(await (await p.export({slide,format:'png',scale:1})).arrayBuffer()));
 const layout=await slide.export({format:'layout'});await fs.writeFile(`${build}/rendered/slide-${String(i+1).padStart(2,'0')}.json`,await layout.text());
}
await fs.writeFile(build+'/montage.webp',new Uint8Array(await (await p.export({format:'webp',montage:true,scale:.5})).arrayBuffer()));
console.log('Rendered',p.slides.items.length);
const result=await finalizePresentation({workspaceDir:work,candidatePath:build+'/draft.pptx',finalPath:final,pythonExecutable:'C:/Users/sherl/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','2','--require-native-table-slide','10'],explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[2,10],fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:build+'/validation.json'});
console.log(JSON.stringify(result));

