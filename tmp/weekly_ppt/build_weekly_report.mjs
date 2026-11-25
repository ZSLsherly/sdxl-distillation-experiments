import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "C:/Users/sherl/Documents/ChatGPT/推理加速";
const SKILL_DIR = "C:/Users/sherl/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.11809/skills/presentations";
const TMP_DIR = path.join(workspaceDir, "tmp/weekly_ppt");
const FINAL_PPTX = path.join(workspaceDir, "output/pptx/扩散模型推理加速周报_浅色对齐版_v3.pptx");
const RUNTIME_PYTHON = "C:/Users/sherl/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe";
const { finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });

const W = 1280;
const H = 720;
const FONT = "Microsoft YaHei";
const C = {
  bg: "#F5F8FC",
  panel: "#FFFFFF",
  panel2: "#EAF2FA",
  white: "#172436",
  muted: "#5C6F84",
  line: "#CBD7E4",
  cyan: "#159DB2",
  blue: "#3778D8",
  violet: "#8059C8",
  orange: "#D98525",
  green: "#23986F",
  red: "#D95863",
};

const p = Presentation.create({ slideSize: { width: W, height: H } });

function shape(slide, x, y, w, h, fill = "none", radius = "rect", line = "none") {
  return slide.shapes.add({
    geometry: radius,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: line === "none" ? { fill: "none", width: 0 } : { fill: line, width: 1 },
  });
}

function text(slide, value, x, y, w, h, size = 24, color = C.white, options = {}) {
  const box = slide.shapes.add({
    geometry: "textbox",
    position: { left: x, top: y, width: w, height: h },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  box.text = value;
  box.text.style = {
    typeface: FONT,
    fontSize: size,
    color,
    bold: Boolean(options.bold),
    alignment: options.align ?? "left",
    verticalAlignment: options.valign ?? "middle",
    autoFit: options.autoFit ?? "shrinkText",
  };
  return box;
}

function addNotes(slide, body, sources = []) {
  const refs = sources.length ? `\n\n参考资料：\n${sources.join("\n")}` : "";
  slide.speakerNotes.textFrame.setText(body + refs);
}

function newSlide() {
  const slide = p.slides.add();
  slide.background.fill = C.bg;
  return slide;
}

function header(slide, title, subtitle, number) {
  shape(slide, 0, 0, 12, H, C.cyan);
  text(slide, title, 60, 32, 1050, 58, 38, C.white, { bold: true });
  if (subtitle) text(slide, subtitle, 62, 91, 1080, 30, 18, C.muted);
  text(slide, String(number).padStart(2, "0"), 1170, 42, 52, 34, 17, C.muted, { align: "right" });
}

function tag(slide, value, x, y, w, color) {
  shape(slide, x, y, w, 34, color, "roundRect");
  text(slide, value, x + 8, y + 2, w - 16, 30, 16, C.bg, { bold: true, align: "center" });
}

function methodSlide({ number, title, subtitle, problem, method, result, formula, color, notes, sources }) {
  const slide = newSlide();
  header(slide, title, subtitle, number);
  tag(slide, "核心问题", 64, 154, 124, color);
  text(slide, problem, 64, 206, 500, 148, 25, C.white, { bold: true, valign: "top" });
  shape(slide, 618, 154, 2, 430, C.line);
  tag(slide, "方法", 674, 154, 90, color);
  text(slide, method, 674, 206, 520, 148, 23, C.white, { valign: "top" });
  text(slide, "关键公式", 64, 400, 180, 32, 20, color, { bold: true });
  text(slide, "本页结论", 674, 400, 180, 32, 20, color, { bold: true });
  shape(slide, 64, 446, 500, 130, C.panel, "roundRect", C.line);
  text(slide, formula, 86, 466, 456, 90, 24, color, { bold: true, align: "center" });
  text(slide, result, 674, 446, 510, 130, 24, C.white, { bold: true, valign: "top" });
  text(slide, "周报汇报建议：先讲问题，再讲方法，公式只解释变量关系。", 64, 636, 1120, 26, 17, C.muted);
  addNotes(slide, notes, sources);
}

// 1. Cover
{
  const s = newSlide();
  shape(s, 0, 0, 18, H, C.cyan);
  text(s, "扩散模型推理加速周报", 78, 132, 1030, 80, 54, C.white, { bold: true });
  text(s, "DMD 系列论文学习与下一步复现计划", 82, 226, 970, 48, 30, C.cyan, { bold: true });
  shape(s, 82, 314, 880, 2, C.line);
  text(s, "本周完成：6 篇论文主线梳理、关键公式推导、SenseFlow 教学代码", 82, 348, 990, 48, 22, C.white);
  text(s, "汇报日期  2026.09.16", 82, 604, 420, 32, 18, C.muted);
  addNotes(s, "开场建议控制在 20 秒。说明本周目标是建立 DMD 系列的统一理解，并为下一阶段复现确定可执行路线。");
}

// 2. Map
{
  const s = newSlide();
  header(s, "论文主线", "六篇工作围绕目标函数、训练稳定性和多步扩展逐步推进", 2);
  const items = [
    ["DMD", "一步分布匹配", C.blue],
    ["DMD2", "稳定训练与多步", C.green],
    ["f-Distill", "一般 f 散度", C.violet],
    ["ADM / DMDX", "对抗差异度量", C.orange],
    ["SenseFlow", "大型 Flow 扩展", C.cyan],
    ["Phased DMD", "分阶段多步 MoE", C.red],
  ];
  items.forEach((item, i) => {
    const x = 54 + i * 202;
    shape(s, x, 218, 174, 170, C.panel, "roundRect", item[2]);
    const nameSize = item[0].length > 9 ? 18 : item[0].length > 7 ? 21 : 25;
    text(s, item[0], x + 8, 240, 158, 42, nameSize, item[2], { bold: true, align: "center" });
    text(s, item[1], x + 14, 298, 146, 60, 19, C.white, { bold: true, align: "center" });
    if (i < items.length - 1) text(s, "›", x + 176, 271, 24, 54, 34, C.muted, { bold: true, align: "center" });
  });
  shape(s, 114, 458, 1052, 104, C.panel2, "roundRect");
  text(s, "统一框架", 142, 476, 150, 30, 20, C.cyan, { bold: true });
  text(s, "学生生成样本，教师估计目标分布，fake score 跟踪学生分布，二者差异形成训练信号。", 142, 512, 958, 34, 24, C.white, { bold: true });
  addNotes(s, "这页用于交代论文之间的关系。不要逐篇展开，只说明后续每页会介绍一项关键改进。");
}

// 3-8. Papers
methodSlide({
  number: 3,
  title: "DMD：一步分布匹配",
  subtitle: "One-step Diffusion with Distribution Matching Distillation，CVPR 2024",
  problem: "教师需要多步采样\n如何训练一步学生，同时避免逐轨迹模仿？",
  method: "学生先生成样本，再随机加噪。教师 score 描述目标分布，fake score 描述学生分布。两者差异通过学生 Jacobian 更新生成器。",
  result: "学生匹配教师的输出分布，不必复现同一条采样轨迹。",
  formula: "梯度  ∝  −(s_real − s_fake) · ∂G/∂θ",
  color: C.blue,
  notes: "强调 DMD 的核心是 distribution matching，而不是 trajectory matching。原始 DMD 还使用教师生成的 noise-image pairs 回归损失稳定训练。",
  sources: ["Yin et al., One-step Diffusion with Distribution Matching Distillation, CVPR 2024"],
});

methodSlide({
  number: 4,
  title: "DMD2：训练稳定性与多步学生",
  subtitle: "Improved Distribution Matching Distillation，NeurIPS 2024",
  problem: "原始 DMD 依赖昂贵的离线回归数据，fake score 也可能跟不上学生分布。",
  method: "取消回归损失。提高 fake score 的更新频率，引入真实数据 GAN loss，并用 backward simulation 覆盖多步推理时的输入分布。",
  result: "TTUR、GAN 和 backward simulation 共同支撑无回归数据的少步蒸馏。",
  formula: "fake score 多次更新  +  generator 一次更新",
  color: C.green,
  notes: "可以把 DMD2 讲成三个工程改进。TTUR 修正 fake score 跟踪误差，GAN 提供真实数据约束，backward simulation 缩小训练与推理输入差异。",
  sources: ["Yin et al., Improved Distribution Matching Distillation for Fast Image Synthesis, NeurIPS 2024"],
});

methodSlide({
  number: 5,
  title: "f-Distill：散度选择决定样本权重",
  subtitle: "One-step Diffusion Models with f-Divergence Distribution Matching",
  problem: "DMD 隐式使用 reverse KL，容易偏向少数高概率模式。",
  method: "将 DMD 推广到 f 散度。判别器估计密度比 r=p_teacher/p_student，再用 h(r)=f''(r)r² 给 score difference 动态加权。",
  result: "DMD 是 h(r)=1 的特例，forward KL 与 JS 提供不同的模式覆盖偏好。",
  formula: "梯度  ∝  −h(r)(s_real − s_fake) · ∂G/∂θ",
  color: C.violet,
  notes: "只解释权重含义。r 很大表示教师密度高而学生覆盖少。Forward KL 使用 h(r)=r，会更重视这类区域。",
  sources: ["Xu, Nie, Vahdat, One-step Diffusion Models with f-Divergence Distribution Matching"],
});

methodSlide({
  number: 6,
  title: "ADM 与 DMDX：学习分布差异",
  subtitle: "Adversarial Distribution Matching for Diffusion Distillation",
  problem: "预先指定一种散度可能限制复杂图像与视频分布的匹配能力。",
  method: "ADM 用扩散骨干上的判别头比较 real 与 fake 的局部 PF-ODE 预测。一步模型先经过 ADP，对教师 ODE pairs 做 latent 与 pixel 对抗预训练，再进入 ADM 微调。",
    result: "DMDX 由 ADP 初始化和 ADM 微调组成，教师只提供多步终点和局部方向",
  formula: "DMDX  =  ADP 预训练  +  ADM 微调",
  color: C.orange,
  notes: "避免说教师学会跨步。跨步能力属于学生。ADP 改善一步学生与教师分布的初始重叠，ADM 再匹配不同噪声层级的概率流。",
  sources: ["Lu et al., Adversarial Distribution Matching for Diffusion Distillation, ICCV 2025"],
});

methodSlide({
  number: 7,
  title: "SenseFlow：扩展到大型 Flow 模型",
  subtitle: "Scaling Distribution Matching for Flow-based Text-to-Image Distillation",
  problem: "在 SD3.5 和 FLUX 等大型模型上，单纯提高 TTUR 仍可能震荡且成本很高。",
  method: "IDA 在学生更新后软对齐 fake model。ISG 让稀疏推理锚点吸收区间内教师知识。冻结的 CLIP 或 DINOv2 骨干为 GAN 提供语义与视觉特征。",
  result: "稳定性的关键是减少 fake model 滞后，并改善稀疏时间点的监督质量。",
  formula: "IDA：φ_fake ← λφ_fake + (1−λ)θ_student",
  color: C.cyan,
  notes: "可以展示本周完成的中文教学代码。说明 IDA 不替代 fake model 的 denoising training，只提供学生更新后的快速校正。",
  sources: ["Ge et al., SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation, ICLR 2026"],
});

methodSlide({
  number: 8,
  title: "Phased DMD：分阶段的少步专家",
  subtitle: "Few-step Distribution Matching Distillation via Score Matching within Subintervals",
  problem: "直接反传多步计算图成本高，随机截断又可能把少步模型退化成一步模型。",
  method: "按 SNR 划分阶段，每阶段只训练一个专家并匹配中间分布。上一阶段专家冻结。论文进一步推导带噪起点 x_s 下的无偏子区间 score-matching target。",
  result: "每个时间专家只负责一段生成任务，形成由 timestep 自然路由的 MoE。",
  formula: "第 k 阶段：p_fake(x_tk)  ≈  p_real(x_tk)",
  color: C.red,
  notes: "强调子区间目标的必要性。x_s 已经带噪，不能继续把 ε-x_s 当作普通 flow target，否则 fake score 会产生偏差。",
  sources: ["Fan et al., Phased DMD: Few-step Distribution Matching Distillation via Score Matching within Subintervals, CVPR 2026"],
});

// 9. Comparison
{
  const s = newSlide();
  header(s, "六篇论文的分工", "同一 DMD 骨架上的不同改进层次", 9);
  const cols = [64, 244, 604, 920];
  ["论文", "主要矛盾", "新增机制", "适合回答的问题"].forEach((v, i) =>
    text(s, v, cols[i], 150, [160, 340, 300, 290][i], 38, 19, C.cyan, { bold: true })
  );
  const rows = [
    ["DMD", "一步分布匹配", "real/fake score", "一步学生怎么训练"],
    ["DMD2", "训练成本与稳定性", "TTUR、GAN、backward simulation", "怎样去掉离线回归"],
    ["f-Distill", "reverse KL 的模式偏好", "density ratio 权重", "怎样更换散度"],
    ["ADM / DMDX", "固定散度的表达能力", "学习式判别器、ADP", "怎样做对抗分布匹配"],
    ["SenseFlow", "大模型上的 fake lag", "IDA、ISG、VFM GAN", "怎样扩展到 SD3.5 和 FLUX"],
    ["Phased DMD", "多步反传与一步退化", "中间分布、时间专家", "怎样保留少步容量"],
  ];
  rows.forEach((row, r) => {
    const y = 198 + r * 68;
    if (r % 2 === 0) shape(s, 52, y - 5, 1170, 62, C.panel, "roundRect");
    row.forEach((v, i) => text(s, v, cols[i], y, [160, 340, 300, 290][i], 48, i === 0 ? 19 : 17, i === 0 ? C.white : C.muted, { bold: i === 0 }));
  });
  addNotes(s, "这页用于回答学长追问：六篇文章不是六套互不相关的方法。它们分别修改分布目标、训练动态、判别监督和多步结构。");
}

// 10. Weekly outputs
{
  const s = newSlide();
  header(s, "本周学习产出", "从阅读论文转向可解释、可实现的训练流程", 10);
  const blocks = [
    ["理论基础", "学习 diffusion 基础知识，统一理解数学概念与推导方式。", C.blue],
    ["论文理解", "对各论文的训练目标和创新点形成基础认识。", C.violet],
    ["代码", "完成 SenseFlow 最小教学代码，包含 DMD、ISG、IDA 和 VFM-GAN。", C.green],
    ["不足", "尚未完成真实模型数值复现，当前环境与可用算力需要先审计。", C.orange],
  ];
  blocks.forEach((b, i) => {
    const y = 152 + i * 112;
    text(s, b[0], 72, y, 126, 70, 25, b[2], { bold: true, align: "center" });
    shape(s, 218, y, 2, 70, b[2]);
    text(s, b[1], 254, y, 930, 70, 24, C.white, { bold: i === 3 });
  });
  shape(s, 72, 610, 1110, 2, C.line);
  text(s, "当前判断：论文主线已经完整，下一阶段的价值来自复现、测量和实验记录。", 72, 628, 1100, 38, 23, C.cyan, { bold: true });
  addNotes(s, "这里要如实区分理解和验证。代码完成语法检查不等于完成 PyTorch 数值实验，更不等于真实 SDXL 或 FLUX 复现。");
}

// 11. Plan
{
  const s = newSlide();
  header(s, "未来四周计划建议", "先确认资源，再形成可复现结果与求职作品", 11);
  const weeks = [
    ["第 1 周", "资源审计与基线", "确认 GPU、显存和环境；跑通官方推理；记录 NFE、延迟和显存。", C.blue],
    ["第 2 周", "小规模训练闭环", "在二维数据或小型图像集复现 DMD；验证 fake score 跟踪和 TTUR。", C.green],
    ["第 3 周", "一项可控消融", "比较 DMD 与 f-Distill 权重，或比较 DMD2 有无 backward simulation。", C.violet],
    ["第 4 周", "整理可展示结果", "形成代码仓库、实验表、失败记录和 5 页技术总结，面向实习投递。", C.orange],
  ];
  weeks.forEach((w, i) => {
    const x = 54 + i * 302;
    tag(s, w[0], x, 158, 94, w[3]);
    text(s, w[1], x, 210, 268, 46, 24, C.white, { bold: true });
    shape(s, x, 270, 270, 220, C.panel, "roundRect", w[3]);
    text(s, w[2], x + 20, 290, 230, 176, 21, C.muted);
  });
  text(s, "需要和学长确认", 64, 554, 230, 32, 20, C.red, { bold: true });
  text(s, "可用算力、优先复现的方法、评价数据规模，以及更偏研究还是偏实习作品。", 64, 594, 1110, 46, 23, C.white, { bold: true });
  addNotes(s, "这页是讨论稿。推荐优先做小规模但完整的实验闭环，不直接尝试训练 SD3.5 或 FLUX。求职价值来自可复现代码、性能测量、消融分析和清晰的失败记录。该建议结合此前对个人背景的评估：概率模型、模型架构、CV、LLM 训练与工程经验可以迁移到生成模型推理加速。",
    ["此前会话：评估推理加速的就业价值；本地学习总结与简历匹配分析"]);
}

const requirements = {
  explicitTotalSlideCount: 11,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
};
const fontPolicy = { basis: "design", families: [FONT] };
const expectedSlideSizeEmu = "12192000,6858000";
const stagingDir = path.join(workspaceDir, ".codex-finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const candidatePath = path.join(stagingDir, "weekly_report_candidate.pptx");
await (await PresentationFile.exportPptx(p)).save(candidatePath);

await finalizePresentation({
  ...requirements,
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: [
    "--expected-slide-size-emu", expectedSlideSizeEmu,
    "--validate-bullet-geometry",
    "--validate-heading-fit",
  ],
  requiredNativeTableOwnerSlides: [],
  fontPolicy,
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "weekly_report_light_aligned_v3.validation.json"),
});

console.log(FINAL_PPTX);
