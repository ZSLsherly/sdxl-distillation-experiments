import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:/Users/sherl/Documents/ChatGPT/推理加速";
const SKILL_DIR = "C:/Users/sherl/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const TMP_DIR = path.join(workspaceDir, "tmp/pptx_build");
const FINAL_PPTX = path.join(workspaceDir, "output/pptx/Diffusion与DMD入门学习课件.pptx");
const RUNTIME_PYTHON = "C:/Users/sherl/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const { makeNativeBulletParagraphs, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

const W = 1280, H = 720;
const FONT = "Microsoft YaHei";
const C = {
  navy: "#10253F", blue: "#246BFD", cyan: "#5CC8FF", pale: "#EAF3FF",
  orange: "#F28C28", peach: "#FFF0DF", ink: "#17212B", grey: "#627083",
  light: "#F7F9FC", line: "#D8E1EC", white: "#FFFFFF", green: "#1F9D72",
  red: "#D95D5D", purple: "#6857D9"
};

const p = Presentation.create({ slideSize: { width: W, height: H } });

function rect(slide, x, y, w, h, fill, radius="rect", line="none") {
  return slide.shapes.add({ geometry: radius, position: { left:x, top:y, width:w, height:h }, fill,
    line: line === "none" ? { fill:"none", width:0 } : { fill:line, width:1 } });
}

function text(slide, value, x, y, w, h, size=26, color=C.ink, opts={}) {
  const s = slide.shapes.add({ geometry:"textbox", position:{left:x,top:y,width:w,height:h},
    fill:"none", line:{fill:"none",width:0} });
  s.text = value;
  s.text.style = {
    typeface: FONT, fontSize:size, color, bold:!!opts.bold,
    autoFit: opts.autoFit ?? "shrinkText", alignment: opts.align ?? "left",
    verticalAlignment: opts.valign ?? "middle",
  };
  return s;
}

function bullets(slide, items, x, y, w, h, size=25, color=C.ink, opts={}) {
  const s = slide.shapes.add({ geometry:"textbox", position:{left:x,top:y,width:w,height:h},
    fill:"none", line:{fill:"none",width:0} });
  s.text = makeNativeBulletParagraphs(items, {
    marginLeftPoints: opts.marginLeft ?? 20,
    hangingPoints: opts.hanging ?? 10,
    spaceAfterPoints: opts.spaceAfter ?? 10,
  });
  s.text.style = { typeface:FONT, fontSize:size, color, autoFit:"shrinkText", verticalAlignment:"top" };
  return s;
}

function title(slide, t, sub="") {
  text(slide, t, 68, 34, 1138, 62, 42, C.navy, {bold:true});
  rect(slide, 68, 105, 88, 6, C.orange, "roundRect");
  if (sub) text(slide, sub, 176, 88, 1015, 34, 19, C.grey);
}

function footer(slide, n, section) {
  text(slide, section, 68, 680, 440, 24, 16, C.grey);
  text(slide, String(n).padStart(2,"0"), 1160, 680, 48, 24, 16, C.grey, {align:"right"});
}

function addNotes(slide, body, sources=[]) {
  const refs = sources.length ? `\n\n参考资料：\n${sources.join("\n")}` : "";
  slide.speakerNotes.textFrame.setText(body + refs);
}

function newSlide(bg=C.white) {
  const s = p.slides.add();
  s.background.fill = bg;
  return s;
}

function addImage(slide, file, x, y, w, h, fit="contain", alt="") {
  return slide.images.add({ blob: awaitableRead(file), contentType:"image/png", alt, fit,
    position:{left:x,top:y,width:w,height:h} });
}

function awaitableRead(file) {
  throw new Error("awaitableRead placeholder should be replaced before use");
}

const imgNoise = await fs.readFile(path.join(workspaceDir,"assets/diffusion_course/forward_noise.png"));
const imgScore = await fs.readFile(path.join(workspaceDir,"assets/diffusion_course/score_landscape.png"));
const imgDmd = await fs.readFile(path.join(workspaceDir,"assets/diffusion_course/dmd_distribution.png"));
function image(slide, bytes, x,y,w,h,fit="contain",alt="") {
  return slide.images.add({ blob:bytes, contentType:"image/png", alt, fit, position:{left:x,top:y,width:w,height:h} });
}

// 1 Cover
{
  const s = newSlide(C.navy);
  rect(s, 0, 0, 22, H, C.orange);
  text(s, "Diffusion 与 DMD 入门", 82, 132, 920, 92, 62, C.white, {bold:true});
  text(s, "从普通计算机视觉到生成模型蒸馏", 86, 232, 820, 48, 31, "#CDE5FF");
  text(s, "直觉优先｜每页最多一个关键公式｜配套 4 周学习路线", 88, 312, 890, 38, 23, "#A9BED5");
  rect(s, 88, 408, 850, 2, "#38516E");
  text(s, "适合背景：普通 CV、模型训练、概率统计基础", 88, 438, 760, 36, 22, C.white);
  text(s, "目标：能读懂 DMD 方法图、训练循环和主要改进", 88, 484, 820, 36, 22, C.white);
  text(s, "2026 · 自学课件", 88, 630, 300, 28, 17, "#8197B0");
  addNotes(s, "这套课件按学习顺序组织。第一遍只需理解图、变量和训练数据流。公式用于压缩直觉，不要求先证明。");
}

// 2 How to use
{
  const s = newSlide(); title(s,"这套课件怎么学","先形成闭环，再补推导");
  text(s,"第一遍",80,164,220,42,28,C.blue,{bold:true});
  text(s,"看图和数据流",80,210,300,42,32,C.navy,{bold:true});
  text(s,"知道每个变量是什么，模型输入输出是什么",80,258,360,72,22,C.grey);
  text(s,"第二遍",472,164,220,42,28,C.orange,{bold:true});
  text(s,"读关键公式",472,210,300,42,32,C.navy,{bold:true});
  text(s,"只推 DDPM 加噪、score 转换和 DMD 梯度结构",472,258,360,72,22,C.grey);
  text(s,"第三遍",864,164,220,42,28,C.green,{bold:true});
  text(s,"配小实验",864,210,300,42,32,C.navy,{bold:true});
  text(s,"用可视化确认直觉，再进入完整论文",864,258,340,72,22,C.grey);
  rect(s,80,390,1120,165,C.light,"roundRect");
  text(s,"学习底线",110,414,180,34,24,C.red,{bold:true});
  text(s,"遇到公式时，先回答：它在比较谁？对谁求梯度？这个量由哪个网络给出？",110,458,1030,54,28,C.ink,{bold:true});
  text(s,"暂时允许跳过 ELBO 完整证明、SDE 严格理论和大规模工程细节。",110,518,980,32,21,C.grey);
  footer(s,2,"学习方法"); addNotes(s,"建议把整套课件分三遍使用。第一遍约 60 分钟，第二遍配纸笔，第三遍配代码。不要在第一页公式卡住后停止。所有后续概念都会回到变量、网络角色和训练循环。");
}

// 3 CV bridge
{
  const s = newSlide(); title(s,"普通 CV 与 Diffusion 的接口","网络仍然做监督学习，任务定义发生了变化");
  text(s,"普通 CV",86,158,300,42,30,C.blue,{bold:true});
  text(s,"图片",88,225,160,44,28,C.navy,{bold:true});
  text(s,"分类器 / 检测器",250,225,250,44,25,C.navy,{bold:true});
  text(s,"类别 / 框 / 特征",510,225,240,44,25,C.navy,{bold:true});
  rect(s,88,286,662,3,C.blue);
  text(s,"Diffusion 训练",86,356,300,42,30,C.orange,{bold:true});
  text(s,"带噪图片 + t",88,425,190,44,27,C.navy,{bold:true});
  text(s,"U-Net / DiT",302,425,190,44,25,C.navy,{bold:true});
  text(s,"噪声 / 原图 / 速度",510,425,250,44,25,C.navy,{bold:true});
  rect(s,88,486,662,3,C.orange);
  rect(s,826,158,360,380,C.pale,"roundRect");
  text(s,"你已经会的部分",860,184,290,38,27,C.navy,{bold:true});
  bullets(s,["backbone 与条件编码","损失、反传与优化器","数据增强和噪声建模","训练监控与消融"],850,244,310,230,24,C.ink,{spaceAfter:13});
  text(s,"新增：时间条件 + 迭代生成",860,488,290,34,22,C.orange,{bold:true});
  footer(s,3,"第一章｜Diffusion 最小基础");
  addNotes(s,"普通 CV 的知识没有失效。Diffusion 仍然是监督训练，只是监督标签由程序自动构造。最大变化发生在推理：模型需要多次调用，逐步修正一个随机状态。");
}

// 4 Forward noise image
{
  const s = newSlide(); title(s,"前向加噪","任意噪声等级的训练样本都能一步构造");
  image(s,imgNoise,54,130,1172,392,"contain","一辆蓝色玩具车逐渐被高斯噪声淹没的五阶段示意图");
  rect(s,86,535,1108,98,C.light,"roundRect");
  text(s,"xₜ = αₜ x₀ + σₜ ε",112,548,420,58,36,C.navy,{bold:true,align:"center"});
  text(s,"x₀：干净图像    ε：已知高斯噪声    t：损坏程度",570,550,590,54,23,C.ink);
  text(s,"训练时直接采样 t，无需真的连续加噪 t 次",570,600,590,26,19,C.grey);
  footer(s,4,"第一章｜Diffusion 最小基础");
  addNotes(s,"把 t 暂时理解为图片的损坏程度。αₜ 决定保留多少原图，σₜ 决定加入多少噪声。训练时我们知道 x₀、t 和 ε，因此拥有完整监督信号。",["Ho et al., Denoising Diffusion Probabilistic Models, arXiv:2006.11239","本页概念图由图像生成工具制作，用于教学示意"]);
}

// 5 Reverse denoise
{
  const s = newSlide(); title(s,"反向去噪","同一个网络在不同 t 上完成许多次局部修正");
  const stages=[
    ["高噪声","决定布局","随机状态中形成大色块"],
    ["中噪声","形成主体","轮廓、姿态和颜色出现"],
    ["低噪声","补充细节","纹理、边缘和局部结构"],
  ];
  stages.forEach((a,i)=>{
    const x=86+i*382;
    rect(s,x,176,330,248,i===0?"#EAF3FF":i===1?"#F1EEFF":"#FFF0DF","roundRect");
    text(s,a[0],x+26,198,278,38,27,i===0?C.blue:i===1?C.purple:C.orange,{bold:true});
    text(s,a[1],x+26,252,278,44,31,C.navy,{bold:true});
    text(s,a[2],x+26,318,276,66,22,C.grey);
  });
  text(s,"xₜ   →   xₜ₋Δt   →   …   →   x₀",168,478,944,72,40,C.navy,{bold:true,align:"center"});
  text(s,"一次调用只解决一个局部问题；传统模型需要几十次网络调用",180,555,920,40,24,C.grey,{align:"center"});
  footer(s,5,"第一章｜Diffusion 最小基础");
  addNotes(s,"高噪声阶段负责全局结构，低噪声阶段负责细节，这是后面 Phased DMD 和按 SNR 划分专家的直觉基础。DMD 的目标是让学生一次跨越更长区间。");
}

// 6 parameterizations
{
  const s = newSlide(); title(s,"模型究竟预测什么","四种参数化表达相近的去噪信息");
  const rows=[
    ["ε 预测","找出加入的噪声","需要删除什么",C.blue],
    ["x₀ 预测","直接猜干净图像","最终答案是什么",C.green],
    ["v 预测","给出状态变化方向","下一步怎样移动",C.purple],
    ["score","指向更高概率区域","怎样更像数据",C.orange],
  ];
  rows.forEach((r,i)=>{
    const y=158+i*92;
    text(s,r[0],88,y,190,56,28,r[3],{bold:true});
    text(s,r[1],300,y,380,56,25,C.navy,{bold:true});
    text(s,r[2],730,y,410,56,23,C.grey);
    rect(s,88,y+66,1054,1,C.line);
  });
  rect(s,88,548,1054,78,C.navy,"roundRect");
  text(s,"sθ(xₜ,t) ≈ − εθ(xₜ,t) / σₜ",110,558,520,52,31,C.white,{bold:true,align:"center"});
  text(s,"先会查转换关系，不必先背全部推导",670,560,430,48,21,"#CDE5FF",{align:"center"});
  footer(s,6,"第一章｜Diffusion 最小基础");
  addNotes(s,"不同论文可能使用不同参数化。第一遍只要知道它们通常能在已知噪声日程下相互转换。DMD 论文偏爱 score 的语言，但底层模型可能实际预测 ε、x₀ 或 v。",["Salimans & Ho, Progressive Distillation for Fast Sampling of Diffusion Models, arXiv:2202.00512"]);
}

// 7 DDPM training
{
  const s = newSlide(); title(s,"DDPM 训练循环","监督标签由加噪过程自动生成");
  const labels=["真实图像 x₀","采样 t 与 ε","合成 xₜ","U-Net / DiT","预测 ε̂","计算误差"];
  labels.forEach((lab,i)=>{
    const x=60+i*201;
    rect(s,x,190,168,92,i===3?C.navy:(i===5?C.peach:C.pale),"roundRect");
    text(s,lab,x+10,207,148,56,22,i===3?C.white:(i===5?C.orange:C.navy),{bold:true,align:"center"});
    if(i<5) text(s,"→",x+170,211,30,48,28,C.grey,{bold:true,align:"center"});
  });
  text(s,"Lsimple = 𝔼 ‖ ε − εθ(xₜ,t) ‖²",140,350,1000,70,36,C.navy,{bold:true,align:"center"});
  rect(s,194,452,892,128,C.light,"roundRect");
  text(s,"把它当成普通回归任务",226,468,360,38,28,C.blue,{bold:true});
  text(s,"输入：带噪图像 + timestep",226,518,360,30,22,C.ink);
  text(s,"标签：刚刚由程序加入的噪声",640,518,390,30,22,C.ink);
  footer(s,7,"第一章｜Diffusion 最小基础");
  addNotes(s,"第一遍不需要从 ELBO 推到 simplified loss。核心是：训练样本和标签都由我们自己构造，因此不需要人工标注噪声。t 告诉同一个网络当前面对的损坏程度。",["Ho et al., Denoising Diffusion Probabilistic Models, arXiv:2006.11239"]);
}

// 8 score intuition
{
  const s = newSlide(C.navy);
  text(s,"Score：概率地形上的上坡方向",64,34,1148,60,42,C.white,{bold:true});
  image(s,imgScore,0,104,1280,478,"cover","双峰概率密度地形与沿坡向上的方向箭头");
  rect(s,64,586,1152,78,"#0A1A2D","roundRect");
  text(s,"s(x) = ∇ₓ log p(x)",88,596,400,48,34,C.cyan,{bold:true,align:"center"});
  text(s,"当前位置稍微往哪里移动，概率上升最快？",520,598,650,44,25,C.white,{bold:true});
  text(s,"第二章｜Score 直觉",68,680,440,24,16,"#8097B2");
  text(s,"08",1160,680,48,24,16,"#8097B2",{align:"right"});
  addNotes(s,"score 不是图片质量的标量分数，而是与输入同维度的向量。每个像素或 latent 维度都有一个微小调整方向。图中的球代表样本，白箭头代表局部 score。",["Song et al., Score-Based Generative Modeling through Stochastic Differential Equations, arXiv:2011.13456","本页概念图由图像生成工具制作，用于教学示意"]);
}

// 9 DDPM vs DDIM
{
  const s = newSlide(); title(s,"DDPM、DDIM 与蒸馏","换采样器和重新训练学生是两类方法");
  text(s,"DDPM",92,166,250,42,30,C.blue,{bold:true});
  text(s,"很多小步，通常带随机性",92,214,350,38,23,C.ink);
  text(s,"● ● ● ● ● ● ● ● ● ●",92,272,430,42,26,C.blue);
  text(s,"DDIM",92,360,250,42,30,C.purple,{bold:true});
  text(s,"跳过部分时间点，可走确定性轨迹",92,408,410,38,23,C.ink);
  text(s,"●     ●      ●      ●",92,466,430,42,26,C.purple);
  rect(s,606,166,566,348,C.light,"roundRect");
  text(s,"DMD",642,196,180,42,32,C.orange,{bold:true});
  text(s,"重新训练学生，使极少步生成成为训练目标",642,248,472,62,27,C.navy,{bold:true});
  text(s,"DDIM 减步",642,342,178,34,22,C.grey,{bold:true});
  text(s,"原模型 + 新采样路径",842,342,268,34,22,C.ink);
  text(s,"DMD 蒸馏",642,402,178,34,22,C.grey,{bold:true});
  text(s,"教师 + 新学生模型",842,402,268,34,22,C.ink);
  text(s,"加速目标：减少 NFE，即网络调用次数",642,466,470,30,21,C.orange,{bold:true});
  footer(s,9,"第二章｜采样与加速");
  addNotes(s,"DDIM 可以不重新训练，直接更换采样方式，但步数压到 1 到 4 时质量通常明显下降。DMD 用昂贵训练换便宜推理，专门让学生适应大步长。",["Song et al., Denoising Diffusion Implicit Models, arXiv:2010.02502"]);
}

// 10 latent + CFG
{
  const s = newSlide(); title(s,"Latent Diffusion 与 CFG","Stable Diffusion 的两个必要接口");
  text(s,"图像",70,188,130,44,28,C.navy,{bold:true,align:"center"});
  text(s,"VAE 编码",220,188,160,44,25,C.blue,{bold:true,align:"center"});
  text(s,"latent",405,188,130,44,28,C.navy,{bold:true,align:"center"});
  text(s,"Diffusion",560,188,170,44,25,C.orange,{bold:true,align:"center"});
  text(s,"latent",760,188,130,44,28,C.navy,{bold:true,align:"center"});
  text(s,"VAE 解码",914,188,160,44,25,C.blue,{bold:true,align:"center"});
  text(s,"图像",1090,188,120,44,28,C.navy,{bold:true,align:"center"});
  text(s,"→",190,188,34,44,28,C.grey,{align:"center"}); text(s,"→",375,188,34,44,28,C.grey,{align:"center"});
  text(s,"→",530,188,34,44,28,C.grey,{align:"center"}); text(s,"→",730,188,34,44,28,C.grey,{align:"center"});
  text(s,"→",880,188,34,44,28,C.grey,{align:"center"}); text(s,"→",1068,188,28,44,28,C.grey,{align:"center"});
  rect(s,72,292,1136,3,C.line);
  text(s,"Classifier-Free Guidance",82,338,440,42,29,C.orange,{bold:true});
  text(s,"εcfg = εuncond + w(εcond − εuncond)",82,404,640,70,34,C.navy,{bold:true});
  bullets(s,["放大符合文本条件的方向","w 较大时文本对齐常增强，但可能损失自然度和多样性","蒸馏时固定 CFG 会改变学生要匹配的目标分布"],758,340,420,180,22,C.ink,{spaceAfter:11});
  footer(s,10,"第二章｜Stable Diffusion 接口");
  addNotes(s,"Stable Diffusion 的主去噪过程通常发生在 VAE latent，而不是 RGB 像素。CFG 使用同一模型的有条件与无条件输出，放大文本相关方向。读 DMD 论文时，要检查损失究竟计算在 latent、预测量还是解码后的像素。",["Rombach et al., High-Resolution Image Synthesis with Latent Diffusion Models, arXiv:2112.10752","Ho & Salimans, Classifier-Free Diffusion Guidance, arXiv:2207.12598"]);
}

// 11 DMD gateway
{
  const s = newSlide(C.navy);
  text(s,"进入 DMD 前，只需回答六个问题",72,46,1136,62,44,C.white,{bold:true});
  const qs=[
    "xₜ 怎样由 x₀ 一步构造？","网络预测 ε、x₀、v 还是 score？","score 为什么是一个方向？",
    "采样为什么需要多次调用？","DDIM 减步和蒸馏有何不同？","CFG 改变了什么目标分布？"
  ];
  qs.forEach((q,i)=>{
    const col=i%2,row=Math.floor(i/2),x=76+col*590,y=154+row*142;
    text(s,String(i+1).padStart(2,"0"),x,y,62,44,22,C.orange,{bold:true});
    text(s,q,x+72,y-2,485,58,26,C.white,{bold:true});
    rect(s,x+72,y+70,480,2,"#36506F");
  });
  text(s,"能用自己的话回答，就可以开始 DMD，不必先读完所有扩散原论文。",102,612,1074,44,25,"#CDE5FF",{bold:true,align:"center"});
  text(s,"第三章｜DMD",68,680,440,24,16,"#8097B2"); text(s,"11",1160,680,48,24,16,"#8097B2",{align:"right"});
  addNotes(s,"本页可作为第一阶段验收。若其中某个问题答不上来，回看对应页面；不建议此时继续扩展阅读更多采样器。");
}

// 12 distribution matching image
{
  const s = newSlide(); title(s,"DMD 的关键转变","不要求逐题抄老师，只要求一批结果的总体分布接近");
  image(s,imgDmd,18,126,1244,420,"contain","多步蓝色教师与一步橙色学生生成不同样本但对齐总体分布的示意图");
  rect(s,116,548,1048,72,C.navy,"roundRect");
  text(s,"pstudent(x) ≈ pteacher(x)",150,558,420,48,34,C.white,{bold:true,align:"center"});
  text(s,"允许单个样本不一一对应",620,560,480,44,24,"#CDE5FF",{bold:true,align:"center"});
  footer(s,12,"第三章｜DMD 核心思想");
  addNotes(s,"传统轨迹蒸馏常要求同一个噪声对应教师的具体输出。DMD 允许学生产生不同样本，只要整体质量、语义和多样性分布接近教师。一步学生因此不必复制复杂的确定性映射。",["Yin et al., One-step Diffusion with Distribution Matching Distillation, CVPR 2024","本页概念图由图像生成工具制作，用于教学示意"]);
}

// 13 three roles
{
  const s = newSlide(); title(s,"DMD 的三个角色","生成器负责出图，两个 score 负责导航");
  const cols=[
    ["学生生成器 Gθ","噪声 → 一步图像","需要更新",C.orange,C.peach],
    ["Real score","怎样更像教师","教师冻结",C.blue,C.pale],
    ["Fake score","学生现在容易聚集在哪","持续训练",C.green,"#E8F8F1"],
  ];
  cols.forEach((a,i)=>{
    const x=76+i*388;
    rect(s,x,162,346,282,a[4],"roundRect");
    text(s,a[0],x+28,190,290,48,29,a[3],{bold:true,align:"center"});
    text(s,a[1],x+28,266,290,72,25,C.navy,{bold:true,align:"center"});
    text(s,a[2],x+28,366,290,34,22,C.grey,{align:"center"});
  });
  text(s,"学生更新方向  ∝  real score − fake score",166,494,948,68,36,C.navy,{bold:true,align:"center"});
  text(s,"先把公式读成一句话：朝教师喜欢、但学生尚未扎堆的区域移动",192,568,896,36,23,C.grey,{align:"center"});
  footer(s,13,"第三章｜DMD 核心思想");
  addNotes(s,"real score 通常由冻结教师提供。fake score 是另一个扩散模型，从教师初始化后在学生当前样本上训练。符号正负会因论文损失定义不同而变化，但直觉始终是两个 score 的差异。");
}

// 14 reverse KL
{
  const s = newSlide(); title(s,"反向 KL 与模式丢失","学生更怕生成错误内容，对漏掉某些正确模式不够敏感");
  text(s,"教师分布",98,168,260,38,27,C.blue,{bold:true});
  text(s,"猫模式     狗模式",94,230,420,44,28,C.navy,{bold:true});
  text(s,"▂▅█▅▂           ▂▅█▅▂",92,282,500,58,34,C.blue);
  text(s,"学生分布",98,378,260,38,27,C.orange,{bold:true});
  text(s,"只选择狗模式",96,438,420,44,28,C.navy,{bold:true});
  text(s,"                 ▂▅█▅▂",92,490,500,58,34,C.orange);
  rect(s,650,172,498,326,C.light,"roundRect");
  text(s,"DKL(pstudent ‖ pteacher)",686,202,426,56,31,C.navy,{bold:true,align:"center"});
  text(s,"强惩罚",692,298,120,34,22,C.red,{bold:true});
  text(s,"学生样本落入教师低概率区域",824,294,286,44,21,C.ink);
  text(s,"较弱惩罚",692,382,120,34,22,C.grey,{bold:true});
  text(s,"教师存在、但学生完全没覆盖的模式",824,376,286,54,21,C.ink);
  text(s,"结果：质量可能不错，多样性可能下降",674,460,450,30,21,C.orange,{bold:true,align:"center"});
  footer(s,14,"第三章｜DMD 的数学直觉");
  addNotes(s,"这页只要求理解行为，不要求计算积分。因为反向 KL 的期望在学生样本上计算，学生从不访问的教师模式可能缺少足够惩罚。原始 DMD 因此加入配对回归作为安全绳。",["Yin et al., One-step Diffusion with Distribution Matching Distillation, CVPR 2024"]);
}

// 15 original DMD
{
  const s = newSlide(); title(s,"原始 DMD 的两条训练信号","分布匹配提高真实性，配对回归保护结构和覆盖");
  rect(s,82,164,514,294,C.pale,"roundRect");
  text(s,"分布匹配",112,190,454,42,30,C.blue,{bold:true});
  text(s,"real score − fake score",112,254,454,48,30,C.navy,{bold:true});
  bullets(s,["不要求单样本对应","直接改善整体生成分布","反向 KL 仍有 mode-seeking 风险"],112,326,432,114,21,C.ink,{spaceAfter:8});
  rect(s,684,164,514,294,C.peach,"roundRect");
  text(s,"教师配对回归",714,190,454,42,30,C.orange,{bold:true});
  text(s,"同噪声下接近教师输出",714,254,454,48,28,C.navy,{bold:true});
  bullets(s,["稳定训练与大尺度结构","帮助覆盖更多模式","需要离线运行昂贵教师"],714,326,432,114,21,C.ink,{spaceAfter:8});
  text(s,"L = LDMD + λ Lreg",344,510,592,62,38,C.navy,{bold:true,align:"center"});
  footer(s,15,"第三章｜原始 DMD");
  addNotes(s,"原始 DMD 的回归损失使用预计算的教师 noise-image pairs，并可采用 LPIPS。它限制学生完全偏离教师映射，同时也使数据准备昂贵，并限制学生超越教师轨迹。",["Yin et al., One-step Diffusion with Distribution Matching Distillation, CVPR 2024"]);
}

// 16 DMD2
{
  const s = newSlide(); title(s,"DMD2 的三项改造","去掉配对数据后，用更快的 critic 和真实数据保持稳定");
  const items=[
    ["TTUR","Fake score 更新多次\nGenerator 更新一次","让导航图追上移动中的学生",C.blue],
    ["GAN loss","真实图像与学生图像\n交给判别器","补偿教师 score 的误差",C.orange],
    ["Backward simulation","训练后续步时\n使用学生自己的中间状态","缩小训练和推理输入差异",C.green],
  ];
  items.forEach((a,i)=>{
    const x=66+i*405;
    text(s,a[0],x,166,350,42,29,a[3],{bold:true});
    rect(s,x,218,350,130,C.light,"roundRect");
    text(s,a[1],x+22,232,306,92,25,C.navy,{bold:true,align:"center"});
    text(s,a[2],x,374,350,70,21,C.grey,{align:"center"});
  });
  rect(s,112,498,1056,98,C.navy,"roundRect");
  text(s,"φ ← 多次更新 fake score",146,512,430,54,27,C.cyan,{bold:true,align:"center"});
  text(s,"随后 θ ← 更新 generator",704,512,400,54,27,"#FFD0A0",{bold:true,align:"center"});
  text(s,"重点不是固定 5:1，而是内层估计器必须及时追踪外层生成器",260,568,760,26,19,C.white,{align:"center"});
  footer(s,16,"第四章｜DMD2 与稳定训练");
  addNotes(s,"DMD2 观察到，直接移除回归损失会导致亮度等统计量振荡。原因是 fake score 学习一个不断变化的学生分布。论文在 ImageNet 上使用约 5 次 fake score 更新对应 1 次生成器更新，但比例需要按任务调节。",["Yin et al., Improved Distribution Matching Distillation for Fast Image Synthesis, arXiv:2405.14867"]);
}

// 17 f-distill
{
  const s = newSlide(); title(s,"f-distill：换一种分布扣分规则","基本方向仍是 score difference，变化发生在样本权重");
  text(s,"更新 ∝ wf(xₜ) · (real score − fake score)",148,156,984,70,34,C.navy,{bold:true,align:"center"});
  const cols=[
    ["Reverse KL","偏向高质量区域","可能漏模式","梯度较稳",C.blue],
    ["Forward KL","更重视模式覆盖","可能高方差","难度较高",C.orange],
    ["JS","质量与覆盖折中","方差较低","早期可能饱和",C.green],
  ];
  cols.forEach((a,i)=>{
    const x=72+i*400;
    rect(s,x,270,356,250,C.light,"roundRect");
    text(s,a[0],x+20,292,316,42,28,a[4],{bold:true,align:"center"});
    text(s,a[1],x+28,358,300,38,23,C.navy,{bold:true,align:"center"});
    text(s,a[2],x+28,412,300,34,21,C.grey,{align:"center"});
    text(s,a[3],x+28,458,300,34,21,C.grey,{align:"center"});
  });
  text(s,"没有永远最好的散度：覆盖、梯度方差和训练稳定性需要共同权衡",170,554,940,38,24,C.orange,{bold:true,align:"center"});
  footer(s,17,"第四章｜散度选择");
  addNotes(s,"f-distill 使用判别器估计教师与学生的密度比，再由 f-divergence 生成样本权重。它提供统一视角：很多方法共享 score difference，主要区别在于关注哪些样本。",["Xu, Nie, Vahdat, One-step Diffusion Models with f-Divergence Distribution Matching, arXiv:2502.15681"]);
}

// 18 Later methods
{
  const s = newSlide(); title(s,"后续方法在修补哪一部分","都围绕导航质量、追踪速度和跨步难度");
  const rows=[
    ["ADM / DMDX","固定散度不够灵活","用扩散判别器学习差异；一步蒸馏先做对抗预训练",C.purple],
    ["SenseFlow","大模型上 TTUR 又贵又不稳","IDA 轻量同步 fake model；区间内教师引导",C.blue],
    ["Phased DMD","一次匹配完整时间过程太难","按嵌套时间子区间训练；高噪声到低噪声分工",C.orange],
  ];
  rows.forEach((a,i)=>{
    const y=160+i*142;
    text(s,a[0],76,y,235,50,28,a[3],{bold:true});
    text(s,a[1],322,y,320,50,23,C.navy,{bold:true});
    text(s,a[2],678,y-2,510,66,22,C.ink);
    rect(s,76,y+86,1112,2,C.line);
  });
  rect(s,158,574,964,46,C.navy,"roundRect");
  text(s,"导航方向准不准？   导航员跟不跟得上？   一步跨越是否太难？",182,578,916,36,23,C.white,{bold:true,align:"center"});
  footer(s,18,"第四章｜DMD 系列全景");
  addNotes(s,"ADM 让分布差异可学习；SenseFlow 面向 SD3.5 Large、FLUX 等大型 Flow 模型改善稳定性和区间学习；Phased DMD 将匹配限制在时间子区间，并自然形成按 SNR 分工的专家。",["Lu et al., Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis","Ge et al., SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation","Phased DMD supplementary material supplied by the user"]);
}

// 19 Unified map
{
  const s = newSlide(C.navy);
  text(s,"六篇论文可以压缩成一张地图",70,42,1140,60,42,C.white,{bold:true});
  const nodes=[
    ["DMD","双 score + 反向 KL\n回归损失保稳定",86,154,C.blue],
    ["DMD2","TTUR + GAN\n多步闭环训练",442,154,C.orange],
    ["f-distill","散度决定\n样本权重",798,154,C.green],
    ["ADM","学习差异\n替代固定规则",86,394,C.purple],
    ["SenseFlow","大模型同步\n区间内指导",442,394,C.cyan],
    ["Phased DMD","时间子区间\n分阶段专家",798,394,"#FFB65C"],
  ];
  nodes.forEach((a,i)=>{
    rect(s,a[2],a[3],300,150,"#183550","roundRect","#36516C");
    text(s,a[0],a[2]+22,a[3]+18,256,40,29,a[4],{bold:true,align:"center"});
    text(s,a[1],a[2]+22,a[3]+68,256,60,22,C.white,{bold:true,align:"center"});
  });
  text(s,"→",394,200,34,48,28,"#829AB3",{bold:true,align:"center"});
  text(s,"→",750,200,34,48,28,"#829AB3",{bold:true,align:"center"});
  text(s,"→",394,440,34,48,28,"#829AB3",{bold:true,align:"center"});
  text(s,"→",750,440,34,48,28,"#829AB3",{bold:true,align:"center"});
  text(s,"固定散度路线",88,628,300,30,18,"#8197B0");
  text(s,"可学习差异与规模化路线",800,628,390,30,18,"#8197B0",{align:"right"});
  text(s,"19",1160,680,48,24,16,"#8097B2",{align:"right"});
  addNotes(s,"推荐阅读顺序为 DMD、DMD2、f-distill，随后根据兴趣选择 ADM、SenseFlow 和 Phased DMD。后三篇不应在基础薄弱时并行硬啃。");
}

// 20 Four week plan
{
  const s = newSlide(); title(s,"四周学习计划","每周形成一个可以验证的闭环");
  const weeks=[
    ["第 1 周","Diffusion 基础","加噪、预测噪声、完整采样","输出：一组加噪/去噪中间图",C.blue],
    ["第 2 周","Score 与采样","参数化转换、DDIM、KL 方向","输出：二维双峰 score field",C.purple],
    ["第 3 周","DMD 与 DMD2","三模型数据流、TTUR、回归/GAN","输出：玩具 DMD 稳定性实验",C.orange],
    ["第 4 周","论文与作品集","f-distill 及后续路线、实验整理","输出：仓库 + 报告 + 讲解 PPT",C.green],
  ];
  weeks.forEach((a,i)=>{
    const y=142+i*121;
    text(s,a[0],72,y,130,36,24,a[4],{bold:true});
    text(s,a[1],220,y,230,36,26,C.navy,{bold:true});
    text(s,a[2],470,y,420,36,22,C.ink);
    text(s,a[3],914,y,294,48,20,C.grey);
    rect(s,72,y+72,1136,2,C.line);
  });
  rect(s,142,616,996,42,C.peach,"roundRect");
  text(s,"前两周不碰 SDXL 和视频；先让小实验替你消化公式",166,620,948,34,22,C.orange,{bold:true,align:"center"});
  footer(s,20,"第五章｜学习与实践");
  addNotes(s,"这四周的目标不是完整复现大模型论文，而是形成可以解释、运行、诊断的最小闭环。每天建议 1.5 到 2.5 小时，阅读与实验至少各占一半。");
}

// 21 Portfolio experiment
{
  const s = newSlide(); title(s,"第一个作品集项目","二维实验先验证机制，小图像实验再验证工程");
  rect(s,72,158,536,378,C.pale,"roundRect");
  text(s,"实验 A：二维 Gaussian Mixture",104,184,470,44,28,C.blue,{bold:true});
  bullets(s,["画出教师双峰与学生分布","比较只有 real score 与双 score","比较 reverse KL、JS 或 forward KL","记录 mode coverage 与梯度方差"],104,254,454,218,22,C.ink,{spaceAfter:11});
  text(s,"目的：看见 mode collapse，而不是只背定义",104,480,450,34,20,C.blue,{bold:true});
  rect(s,672,158,536,378,C.peach,"roundRect");
  text(s,"实验 B：CIFAR-10 / ImageNet-64",704,184,470,44,28,C.orange,{bold:true});
  bullets(s,["基线：教师、少步采样、一步学生","比较 DMD、DMD + regression、TTUR","测试 fake:generator 更新比例","报告 FID、precision/recall 与失败样本"],704,254,454,218,22,C.ink,{spaceAfter:11});
  text(s,"目的：展示训练闭环与诊断能力",704,480,450,34,20,C.orange,{bold:true});
  text(s,"面试价值来自：你发现了什么现象，怎样验证原因，改动付出了什么代价",146,578,988,40,24,C.navy,{bold:true,align:"center"});
  footer(s,21,"第五章｜学习与实践");
  addNotes(s,"如果算力有限，二维实验也可以成为高质量技术展示，但要有完整可视化和消融。小图像实验应固定随机种子、保存配置，并明确训练成本。");
}

// 22 checklist
{
  const s = newSlide(C.navy);
  text(s,"开始读 DMD 的自测清单",70,40,1140,60,44,C.white,{bold:true});
  const qs=[
    "为什么任意 xₜ 都能一步从 x₀ 构造？",
    "ε 预测怎样转换成 score？",
    "DDIM 减步与 DMD 蒸馏有何区别？",
    "为什么 real score 之外还要 fake score？",
    "反向 KL 为什么可能漏掉模式？",
    "TTUR 解决的是容量问题还是追踪速度问题？",
    "backward simulation 修复了哪种输入偏移？",
    "一步 FID 更低，为什么仍不代表全面超过教师？",
  ];
  qs.forEach((q,i)=>{
    const col=i%2,row=Math.floor(i/2),x=72+col*590,y=142+row*105;
    rect(s,x,y,42,42,i<4?C.blue:C.orange,"ellipse");
    text(s,"✓",x,y,42,42,24,C.white,{bold:true,align:"center"});
    text(s,q,x+58,y-4,492,54,22,C.white,{bold:true});
  });
  rect(s,156,596,968,54,"#183550","roundRect","#36516C");
  text(s,"答不上来就回到对应页面；答得出来就直接进入论文方法与代码。",176,602,928,42,22,"#CDE5FF",{bold:true,align:"center"});
  text(s,"22",1160,680,48,24,16,"#8097B2",{align:"right"});
  addNotes(s,"建议口头回答每题，每题不超过一分钟。能够用自己的语言讲清楚，比复述论文原句更重要。");
}

await fs.mkdir(TMP_DIR,{recursive:true});
await fs.mkdir(path.dirname(FINAL_PPTX),{recursive:true});
const requirements = {
  explicitTotalSlideCount: 22,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = { basis:"design", families:[FONT] };
const stagingDir = path.join(workspaceDir,".codex-finalizer");
await fs.mkdir(stagingDir,{recursive:true});
const candidatePath = path.join(stagingDir,"candidate-diffusion-dmd-course.pptx");
await (await PresentationFile.exportPptx(p)).save(candidatePath);
await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR,"container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR,"container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs:["--expected-slide-size-emu","12192000,6858000","--validate-bullet-geometry","--validate-heading-fit"],
  fontPolicy,
  verifyArtifactToolImport:true,
  receiptPath:path.join(stagingDir,"Diffusion与DMD入门学习课件.validation.json"),
});
console.log(FINAL_PPTX);
