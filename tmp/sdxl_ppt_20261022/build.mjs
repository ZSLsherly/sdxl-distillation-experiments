import fs from 'node:fs/promises';
import {constants as fsConstants} from 'node:fs';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {Presentation, PresentationFile} from 'file:///C:/Users/sherl/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';

const ROOT='C:/Users/sherl/Documents/ChatGPT/推理加速';
const BUILD=path.join(ROOT,'tmp/sdxl_ppt_20261022');
const SKILL='C:/Users/sherl/.codex/plugins/cache/openai-primary-runtime/presentations/26.1017.22135/skills/presentations';
const PY='C:/Users/sherl/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const FINAL=path.join(ROOT,'output/pptx/SDXL_7.5K初步实验_阶段规划与GAN_20261022.pptx');
const FONT='Microsoft YaHei';
const C={ink:'#142E4A',muted:'#637386',blue:'#2474B5',green:'#167965',warm:'#B26B29',light:'#EAF3FA'};
const p=Presentation.create({slideSize:{width:1280,height:720}});
function text(s,value,x,y,w,h,size=26,color=C.ink,bold=false){
 const z=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 z.text=value; z.text.insets=0;
 z.text.style={typeface:FONT,fontSize:size,color,bold,verticalAlignment:'middle',alignment:'left',autoFit:'none'};
 return z;
}
function slide(title,num,sub=''){
 const s=p.slides.add(); s.background.fill='#FFFFFF';
 text(s,title,62,42,1150,70,44,C.ink,true);
 if(sub)text(s,sub,64,118,1140,45,23,C.muted);
 text(s,String(num).padStart(2,'0'),1180,665,45,26,17,C.muted);
 return s;
}
function notes(s,body){s.speakerNotes.textFrame.setText(body);}
const inputNote='来源：用户于2026-10-22提供的两张实验截图。G118数据核对：output/sdxl-crossdistill/benchmark_curves_four_20261018/checkpoint_means.csv，student_updates=118，student_samples=7552，n=4。数值比较按截图展示精度计算。';

{
 const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,'SDXL 少步蒸馏',64,100,1110,90,64,C.ink,true);
 text(s,'阶段规划与 GAN 的初步验证',64,202,1140,76,47,C.ink,true);
 text(s,'7,552 个训练样本，G118 对照',67,328,1090,50,30,C.blue,true);
 text(s,'轨迹变化集中在开局和末端\n末步 GAN+DMD 出现正向信号',67,432,1100,106,32,C.ink);
 text(s,'基础实验汇报    2026.10.22',67,626,1050,34,22,C.muted);
 notes(s,inputNote+'\n本汇报只讨论本轮两组基础实验。7.5K指student_samples=7552，不表示7552次优化更新，也不表示独立训练数据集大小。当前趋势来自固定4张样本均值，未作统计显著性或最终效果声明。');
}
{
 const s=slide('实验一：去噪轨迹在开局和末端变化较大',2,'SDXL native timestep 从左到右减小，按每 20 个时间步统计方向变化');
 const b=await fs.readFile(path.join(BUILD,'turning.png'));
 s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt:'用户提供的SDXL轨迹方向变化柱状图',fit:'contain',position:{left:48,top:184,width:800,height:506}});
 text(s,'观测',892,206,300,44,29,C.blue,true);
 text(s,'开局变化最大\n中段逐渐降低\n末端再次抬升',892,260,310,150,28);
 text(s,'工作假设',892,440,300,44,29,C.blue,true);
 text(s,'开局侧重语义形成\n后期侧重细节调整',892,493,320,96,27);
 text(s,'曲线本身尚不能确定\n语义／细节的阶段边界',892,607,320,64,21,C.muted);
 notes(s,inputNote+'\n曲线是用户截图，保留原始图像和比例。纵轴为每bin平均方向变化角度，不能直接等同于语义或纹理贡献。开局语义、末端细节是本轮工作假设。不同teacher、sampler下需重新验证边界。');
}
{
 const s=slide('实验二：第 4 步 GAN+DMD 的两项指标最好',3,'同样 G118，约 7.5K 个训练样本，指标为 4 张固定样本均值');
 const vals=[
 ['方案','G','耗时 / h','GPU 小时','NIQE ↓','CLIP cosine ↑'],
 ['纯 DMD · 4 卡','118','4.51','18.04','6.0727','0.2980'],
 ['第4步 GAN+DMD · 2卡','118','9.58','19.16','5.9072','0.3053'],
 ['第4步纯 GAN · 2卡','118','7.12','14.24','6.2449','0.3031'],
 ['第3、4步 GAN+DMD · 4卡','118','4.24','16.96','5.9869','0.3000']];
 const t=s.tables.add({rows:5,columns:6,left:60,top:197,width:1160,height:330,columnWidths:[440,70,145,155,160,190],values:vals});
 t.borders.assign({fill:'#D9E2EB',width:1,style:'solid'});
 for(let r=0;r<5;r++)for(let c=0;c<6;c++){
  const cell=t.getCell(r,c);
  cell.fill=r===0?C.ink:r===2?'#EAF5F0':'#FFFFFF';
  cell.text.style={typeface:FONT,fontSize:23,color:r===0?'#FFFFFF':C.ink,bold:r===0||r===2,alignment:c===0?'left':'center',verticalAlignment:'middle'};
 }
 text(s,'相对纯 DMD：NIQE −2.72%，CLIP +2.45%',64,562,1150,46,31,C.green,true);
 text(s,'GPU 小时 +6.2%。卡数不同，耗时不能直接用于判断算法加速。',64,626,1130,40,23,C.muted);
 notes(s,inputNote+'\n表格完整保留截图四个方案及六列数据。NIQE变化=(5.9072/6.0727-1)*100=-2.72696%；CLIP变化=(0.3053/0.2980-1)*100=2.44966%；GPU小时变化=(19.16/18.04-1)*100=6.20843%。NIQE展示值也可按未舍入原始数据得到-2.72412%，幻灯片以原始数值的2.72%为主。GPU小时是卡数乘墙钟时间，包含具体运行环境影响。');
}
{
 const s=slide('GAN 仍有空间，末步联合监督更值得优先验证',4,'以纯 DMD 为参照，当前证据支持继续研究后期 GAN 的加入方式');
 text(s,'第 4 步 GAN+DMD',65,207,490,50,31,C.green,true);
 text(s,'NIQE 下降 2.72%\nCLIP 提高 2.45%',65,277,490,106,35,C.green,true);
 text(s,'当前四组中两项指标均最好\n支持“后期加入 GAN”继续验证',65,414,500,95,27);
 text(s,'第 4 步纯 GAN',690,207,500,50,31,C.warm,true);
 text(s,'NIQE 上升 2.84%\nCLIP 提高 1.70%',690,277,500,106,35,C.warm,true);
 text(s,'CLIP 的变化未带来 NIQE 同向改善\n末步保留 DMD 约束可能更稳妥',690,414,510,95,27);
 text(s,'GAN 扩展到第 3、4 步后，两项指标均弱于仅第 4 步联合。',65,572,1140,48,28,C.ink,true);
 text(s,'小样本结果尚不足以认定最优分工或稳定收益。',65,640,1120,32,22,C.muted);
 notes(s,inputNote+'\n原始数据比较：joint4相对dmd_only的NIQE下降约2.724%，CLIP上升约2.451%。gan4相对dmd_only的NIQE上升约2.835%，CLIP上升约1.699%，按截图四位小数为1.71%。第3、4步联合的GPU小时更低，不能因此判定所有维度上受末步联合支配。GAN对细节的因果改善仍需固定样本细节裁剪或局部指标验证。');
}
{
 const s=slide('当前判断与下一轮验证',5);
 text(s,'当前判断',65,183,1110,50,31,C.blue,true);
 text(s,'开局语义、后期细节的规划值得继续验证\nGAN 在后期仍有发挥空间，优先保留末步 GAN+DMD',65,252,1140,108,32,C.ink,true);
 text(s,'下一轮',65,407,1110,48,31,C.blue,true);
 text(s,'固定教师、初始化和评测 prompts / seeds，复核末步联合的收益',65,478,1140,45,27);
 text(s,'扩大评测样本并重复运行，结合细节裁剪检查纹理和伪影',65,538,1140,45,27);
 text(s,'在相同 GPU 小时预算下比较，检验收益是否稳定',65,598,1140,45,27);
 notes(s,inputNote+'\n此页是由用户判断和当前表格形成的研究方向，尚未被本轮结果证明。后续比较需控制teacher/data/初始化/训练与评测协议/随机种子，并同时报告训练样本数与wall-clock或GPU小时。NIQE与CLIP仅为初步信号，不能替代完整生成质量评测。');
}

// Keep source figures and precise numeric comparisons traceable in local build assets.
await fs.writeFile(path.join(BUILD,'deltas.json'),JSON.stringify({joint4:{niqe_pct:(5.907237933553317/6.072664381364922-1)*100,clip_pct:(0.3053007498383522/0.2979968786239624-1)*100},gan4:{niqe_pct:(6.244946341064246/6.072664381364922-1)*100,clip_pct:(0.30305998027324677/0.2979968786239624-1)*100}},null,2));
for(let i=0;i<p.slides.items.length;i++){
 const blob=await p.export({slide:p.slides.items[i],format:'png',scale:1});
 await fs.writeFile(path.join(BUILD,`slide-${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
const draft=path.join(BUILD,'draft.pptx');
await(await PresentationFile.exportPptx(p)).save(draft);
const {finalizePresentation}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
// This Windows sandbox disallows hard links. Preserve exclusive publication and
// all byte/hash checks using an ordinary file copy only for this exact output.
const originalLink=fs.link;
fs.link=async (source,destination)=>{
 try {return await originalLink(source,destination);}
 catch(error){
  if(error.code!=='EPERM'||path.resolve(destination)!==path.resolve(FINAL))throw error;
  return await fs.copyFile(source,destination,fsConstants.COPYFILE_EXCL);
 }
};
const result=await finalizePresentation({workspaceDir:ROOT,candidatePath:draft,finalPath:FINAL,pythonExecutable:PY,integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit','--require-native-table-slide','3'],explicitTotalSlideCount:5,requiredNativeTableOwnerSlides:[3],fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,receiptPath:path.join(BUILD,'validation.json')});
console.log(JSON.stringify({final:FINAL,result}));
