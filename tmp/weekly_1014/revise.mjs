import fs from 'node:fs/promises';
const dir='C:/Users/sherl/Documents/ChatGPT/推理加速/tmp/weekly_1014';
let s=await fs.readFile(dir+'/build.mjs','utf8');
s=s.replace("const build=work+'/tmp/weekly_1014';","const build=work+'/tmp/weekly_1014_v2';");
s=s.replace('周组会_少步扩散蒸馏论文梳理_20261014.pptx','周组会_少步扩散蒸馏论文梳理_20261014_v2.pptx');
s=s.replace('sCM、rCM、Flash-DMD','rCM（含 sCM）、Flash-DMD');
s=s.replace('覆盖七篇论文','围绕六项主要工作展开，sCM 作为 rCM 的方法基础介绍');
s=s.replace("['一致性训练','sCM、rCM'","['一致性训练','rCM（含 sCM）'");
s=s.replace('sCM 先讲清一致性路线，rCM 再解释它如何与 DMD 联合。','sCM 并入 rCM 页，用作连续时间一致性训练的基础背景。');
const start=s.indexOf("method({title:'sCM：");
const end=s.indexOf("method({title:'AC-DMD：",start);
if(start<0||end<0)throw Error('Missing source section');
s=s.slice(0,start)+`method({title:'rCM：sCM 与分布匹配联合训练',sub:'Large Scale Diffusion Distillation via Score-Regularized Continuous-Time Consistency',color:C.violet,
 problem:'单独使用连续时间一致性训练，\\n大规模图像和视频生成仍可能\\n出现细节失真。',
 method:'一致性分支（sCM）：真实数据加噪，\\n用 JVP 沿教师方向构造一致性监督。\\n分布分支（DMD）：学生生成样本，\\n由 teacher / fake 差异提供监督。\\n两份梯度累积，更新同一个学生。',
 key:'L_student = L_sCM + λ L_DMD\\nsCM 学终点一致性\\nDMD 学生成分布',
 result:'sCM 作为 rCM 的一致性训练基础。\\nDMD 分支补充生成分布监督。\\nfake 单独更新，推理只用学生。',
 footer:'sCM 的 TrigFlow 与稳定训练设计作为基础，JVP 只在训练时使用',
 notes:'先在 rCM 内介绍 sCM 基础：同一条教师轨迹上的不同位置应预测同一终点。真实数据加噪，教师提供局部生成方向，JVP 计算学生输出沿位置和时间共同变化的方向导数，并据此构造连续时间一致性监督。sCM 通过 TrigFlow、目标归一化、自适应权重和预热稳定训练。rCM 保留这条一致性训练流程，再新增 DMD 分支：学生自行生成，teacher 与 fake 的差异提供分布监督。联合阶段两份损失加权求和，或分别反传后累积梯度，再做一次学生更新。交替的是学生更新与 fake 更新。论文随机选择一至四步 rollout，DMD 主要通过最后一次学生预测反传，前面的步骤停止梯度。训练初期有 warmup。sCM 在历史上是独立论文，本次仅作为 rCM 基础介绍，不单列汇报。',sources:[src.rcm,src.scm]});
`+s.slice(end);
s=s.replace("['sCM','连续时间一致性与稳定训练','学生的终点映射'],['rCM','sCM 与 DMD 联合','同一个学生，两份监督']","['rCM（含 sCM）','sCM 一致性与 DMD 联合','同一个学生，两份监督']");
s=s.replace("['sCM','Simplifying, Stabilizing", "['sCM 基础','Simplifying, Stabilizing");
s=s.replace("完整标题与 arXiv 编号，原文链接保存在本页备注","六项工作与 sCM 基础文献，原文链接保存在本页备注");
s=s.replace("'--require-native-table-slide','11'","'--require-native-table-slide','10'");
s=s.replace('explicitTotalSlideCount:12,requiredNativeTableOwnerSlides:[2,11]','explicitTotalSlideCount:11,requiredNativeTableOwnerSlides:[2,10]');
await fs.mkdir('C:/Users/sherl/Documents/ChatGPT/推理加速/tmp/weekly_1014_v2',{recursive:true});
await fs.writeFile('C:/Users/sherl/Documents/ChatGPT/推理加速/tmp/weekly_1014_v2/build.mjs',s);
