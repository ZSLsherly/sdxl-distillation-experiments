from pathlib import Path
from html import escape
import json

OUT = Path(__file__).resolve().parent
SHA = "bacc57f60c4366cb9374b59860fb094e4edb419f"
audit = json.loads((OUT/"repository_audit.json").read_text(encoding="utf8"))

def src(path, line, label=None):
    return f'<a href="https://github.com/byliutao/CDM/blob/{SHA}/{path}#L{line}">{escape(label or f"{path}:{line}")}</a>'

parts = ["""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>CDM 论文与代码对照审查</title>
<style>
body{font-family:system-ui,"Microsoft YaHei",sans-serif;background:#f4f6f9;color:#243248;line-height:1.8;margin:0}
main{max-width:1120px;margin:32px auto;padding:32px 40px;background:#fff;border-radius:12px}
h1{line-height:1.35;font-size:30px}h2{font-size:23px;margin-top:38px;border-bottom:2px solid #dce5ef;padding-bottom:8px}
h3{font-size:18px;margin-top:24px}a{color:#1767ad;text-decoration:none}a:hover{text-decoration:underline}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}td,th{padding:11px;border:1px solid #dbe2ec;text-align:left;vertical-align:top}th{background:#edf2f8}
code{background:#edf2f7;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}pre{white-space:pre-wrap;background:#edf2f7;padding:18px;font-size:14px}
.summary{padding:18px 22px;border-left:5px solid #2778b9;background:#edf5fc}.important{padding:16px 22px;border-left:5px solid #c2782b;background:#fff5e9}
.muted{color:#68768b;font-size:14px}li{margin:8px 0}.wide{overflow-x:auto}
@media(max-width:700px){main{margin:0;padding:20px}h1{font-size:25px}table{font-size:12px}}
</style></head><body><main>
<h1>CDM：论文机制基本对上，默认复现配置与评测需修正</h1>
<p class="muted">审查日期：2026-10-20 · 论文：arXiv 2605.06376v1 · 代码提交：bacc57f60c4366cb9374b59860fb094e4edb419f（2026-05-11）</p>
<div class="summary"><b>结论：</b>公开仓库确实实现了动态时间轨迹、CA、DM 和 off-trajectory CDM；核心损失方向、real/fake 停梯度以及共享轨迹索引可对上。
但它不能直接视为“按 README 启动便能严格复现论文表格”的版本。最值得先处理的是训练/评测提示词大量重叠、FID 两侧预处理不一致、LongCat 训练方式及时间范围配置差异。</div>
<p>完整 Git 仓库已下载至 <code>research_repos/CDM</code>，源码保持未修改。论文原件：<a href="paper.pdf">paper.pdf</a>。
本次完成静态调用路径审查、配置求值、数据文本逐项比对和小型 CPU 数值测试；没有重跑 CDM 完整训练或验证论文模型分数。正在运行的 SDXL COCO 评测保持运行。</p>
<p><a href="https://github.com/byliutao/CDM">官方仓库</a> · <a href="https://arxiv.org/abs/2605.06376">论文入口</a> ·
<a href="repository_audit.json">配置、数据与源码哈希</a> · <a href="cpu_checks.json">CPU 检查结果</a> ·
<a href="overlap_evidence.csv">重叠提示词行号与哈希证据</a></p>
<h2>1. 网络、损失和梯度路径</h2>
<pre>Student θ：负责生成轨迹，以及有梯度的 clean prediction
Real teacher φ：冻结，仅提供条件/无条件预测
Fake teacher ψ：用 student 生成样本做 flow-matching 更新

无梯度 student rollout → 缓存 x_t、v_t、Dθ(x_t,t)
                        ├─ fake：对 Dθ(x_t,t).detach() 加噪，拟合 ε − x0
                        ├─ CA：同一锚点预测 x0 → teacher cond − uncond
                        ├─ DM：同一锚点预测 x0 → teacher real − fake
                        └─ CDM：x_t + (t′−t)·cached_v_t
                                → student 新预测 x0′
                                → x0′.detach() 加噪 → teacher real − fake

CA + DM + CDM → 一次合并 backward → 更新 student
fake optimizer 更新后另有 ψ ← 0.999ψ + 0.001θ</pre>
<p>这里 <code>D(x,t)=x−t·v(x,t)</code>。CDM 是在外推点重新运行 student，再对这个新预测加噪；
不是直接把原锚点的 x0 当作 CDM teacher target，也不是另加像素重建或 GAN。</p>
<table><tr><th>检查项</th><th>实现与论文对应</th><th>源码证据</th></tr>"""]

rows = [
("CA 与 DM 方向", "与式(3)/(4)相符：target = stopgrad(student_x0) + normalized teacher difference；teacher 输出 detach。", src("cdm/training/losses.py",67)+"<br>"+src("cdm/training/losses.py",303)),
("共享锚点", "sd3()/longcat() 均启用 share_latent_indices；fake、CA、DM、CDM 从同一个 per-sample 锚点开始。", src("scripts/train.py",632)+"<br>"+src("config/config.py",485)),
("fake 的 x0_next", "all_x0[0] 是占位初始噪声；all_x0[i+1] 正是锚点 i 的单步 clean prediction。名称容易误读，但这里不是错位取下一锚点。CPU 模拟检查通过。", src("cdm/pipeline/pipeline.py",166)+"<br>"+src("scripts/train.py",647)),
("速度外推", "默认 trajectory_eps：x′ = x + (t′−t)·v；与式(7)对应，默认不是 fresh-noise 替代外推。", src("scripts/train.py",900)),
("CDM 的局部预测目标", "teacher_x0_base='student_pred'，对外推点的新预测 detach 后独立加噪，再取 real−fake；与式(8)/(9)的局部目标对应。", src("config/config.py",490)+"<br>"+src("scripts/train.py",944)),
("梯度隔离", "real teacher no_grad；student loss 中 fake no_grad；fake FM 的目标 detach；三个 student loss 相加后统一 backward。", src("cdm/training/model_utils.py",321)+"<br>"+src("scripts/train.py",1018)),
]
for row in rows:
    parts.append("<tr>"+"".join(f"<td>{v}</td>" for v in row)+"</tr>")
parts.append("""</table>
<p><b>数值验证：</b>直接提取原仓库损失函数执行 CPU autograd，CA/DM 梯度与其解析式最大绝对误差分别约 1.49×10⁻⁸、3.73×10⁻⁹；teacher/fake 预测没有收到 student loss 梯度。
这些是局部函数测试，不代表整个 FSDP 训练已完成运行验证。</p>
<h2>2. 优先关注：公开数据划分</h2>
<div class="important"><b>当前默认训练文本与评测文本存在大量完全相同的条目。</b>比较方式与数据加载器一致：逐行读取并 strip，仅做精确字符串匹配，无模糊匹配或语义推断。</div>
<table><tr><th>评测文件</th><th>条目数</th><th>出现在 mixed_200k/train.txt 的条目</th><th>比例</th></tr>""")
for name in ["pickscore","ocr","mixed_200k","coco2014val_10k"]:
    x=audit["overlap"][name]
    parts.append(f'<tr><td>{name}/test.txt</td><td>{x["test_rows"]:,}</td><td>{x["matching_test_rows"]:,}</td><td>{x["fraction"]:.2%}</td></tr>')
parts.append(f"""</table>
<p>论文 §4.1 使用 PickScore test 提示词评估多项质量指标，附录 C.1 描述从各来源的训练划分抽取训练提示词。
实际默认训练入口读取 <code>mixed_200k/train.txt</code>；评测入口读取各目录 <code>test.txt</code>：
{src("scripts/train.py",319)}、{src("scripts/eval.py",154)}、{src("scripts/eval.py",55)}。</p>
<p><b>影响：</b>用该默认数据重训后，PickScore/CLIP/HPS/Aesthetic 等共享 PickScore prompts 的指标，以及 OCR，
大部分评测提示词不再是未见提示词。这里是<strong>提示词重叠</strong>，不是“真实测试图片参与训练”的证据。
它也不能单独证明作者论文运行使用了相同文件，更不能据此断言论文数值造假；应要求原始数据清单和运行配置。</p>
<p>实际训练文本有 <b>191,500 行</b>（190,360 个不同字符串）；PickScore test 为 2,048 行，OCR 为 1,018 行，DPG 为 1,065 个 item_id。
“200K/2K/1K”可能是约数；与大量 train/test 重叠相比，单纯约数差异不是主要问题。</p>
<h2>3. FID 评测路径有实际可比性问题</h2>
<p>默认 <code>scripts.eval</code> 调用 <code>compute_fid_from_pt_files()</code>。这条路径中：</p>
<ul><li>生成图：浮点 tensor → PyTorch bicubic 缩至 299，<b>未启用 antialias</b> → 乘 255。</li>
<li>真实图：clean-FID <code>get_folder_features(mode="clean")</code> → PIL clean resize。</li></ul>
<p>两侧并未共用相同预处理，也与其另一个“先存 PNG 再算 FID”的路径不同。证据：
{src("scripts/eval.py",681)}、{src("cdm/rewards/fid_scorer.py",298)}、{src("cdm/rewards/fid_scorer.py",393)}、
{src("cdm/rewards/fid_scorer.py",416)}。</p>
<p>用同一张 1024² 高频棋盘图做 CPU 检查，两条 299² 预处理输出的平均绝对差为 <b>49.80/255</b>。
这个例子证明高频缩小时的混叠差异真实存在，<b>不是</b>估算论文 FID 被改变了多少；真实图像的分数影响仍需统一预处理后重算。</p>
<p>此外，仓库 COCO 文件实际为 <b>9,986</b> 行，metadata 也记录 9,986（并标注 resolution=256）。
默认多卡 DistributedSampler 会补齐样本，8 卡对应 9,992 个生成条目；FID 汇总读取全部 batch 文件，未按原样本 ID 去重。
这是小幅但真实的协议偏差。{src("scripts/eval.py",301)}、{src("cdm/rewards/fid_scorer.py",365)}。</p>
<p>与你当前正在跑的 DMD2 官方 COCO10K 对比，两个集合只有 <b>{audit["coco_prompt_comparison"]["exact_shared_unique_prompts"]}</b> 个完全相同的提示词字符串，
因此既不是同一提示词集合，也不是同一 FID 图像预处理。CDM 仓库当前还没有对应的 Patch FID 路径，不能直接拿其论文 FID 与你的新结果横向排名。</p>
<h2>4. 默认配置与论文不完全一致</h2>
<table><tr><th>项目</th><th>论文 / 作者说明</th><th>当前代码实际行为</th><th>影响与证据</th></tr>
<tr><td>LongCat 训练方式</td><td>附录 C.1.2：LoRA，rank 64 / alpha 128</td><td>student.use_lora=False；fake_teacher.use_lora=False</td><td>默认实际全参数训练，显存、优化行为均不同。{src("config/config.py",527)}</td></tr>
<tr><td>动态轨迹内部时间点</td><td>正文以 (0,1] 连续时间描述；作者在 issue #10 称 LongCat 实验下界为 0.02</td><td>sample.random_sigma_min/max 实际为 0.25/0.95；LongCat 将 0.02/0.98 写到错误的顶层字段</td><td>传给原生 scheduler 前的内部采样范围不同，顶层覆盖不生效；原生 scheduler 还可能进行 shift。{src("config/config.py",249)}、{src("config/config.py",545)}、{src("scripts/train.py",608)}</td></tr>
<tr><td>时间边界</td><td>公式常写连续 U(0,1]</td><td>teacher 与 CDM 新时间由 U(0,1) 采样后 clamp 到 [0.02,0.98]</td><td>边界产生点质量；属于未在简式体现的实现细节。不能因此说 CDM 新点也被限在 [0.25,0.95]。{src("cdm/utils/common.py",379)}</td></tr>
<tr><td>weight decay</td><td>附录 C.1：0.001</td><td>0.0001</td><td>10 倍差异，复现需明确选择哪套。{src("config/config.py",61)}</td></tr>
<tr><td>batch 与 GPU 数</td><td>SD3 global batch 128，LongCat 64；16 A100</td><td>每 rank 16 / 8；README 启动 8 进程</td><td>按 README 为 128 / 64；直接改成 16 卡会变为 256 / 128。不能只按论文 GPU 数照搬命令。{src("config/config.py",498)}、{src("config/config.py",550)}、{src("README.md",92)}</td></tr>
<tr><td>训练更新次数</td><td>SD3 4K iterations，LongCat 2K；fake:student=2:1</td><td>SD3：4001 次外层、4001 次 fake、2000 次 student；LongCat：2001/2001/1000</td><td>论文“iterations”是否指 student 更新有歧义，不能把默认 epochs 当成 student 更新次数。{src("scripts/train.py",526)}、{src("scripts/train.py",702)}、{src("scripts/train.py",715)}</td></tr>
</table>
<p>学习率 student=1e−5、fake=5e−6，Adam β=(0.9,0.999)，共享锚点、三个 loss 权重均为 1，
TTUR 2:1 和 Nmax=28 均能对上。应保留这些正确对应，不应把所有差异归为方法实现错误。</p>
<h2>5. 已向公开回复核实的设计与仍需确认的梯度细节</h2>
<h3>CA 受限加噪：有文献与作者解释支持</h3>
<p>CA 默认 <code>uniform_capped</code>：τ = U(0,1)·t，再做边界 clamp。DM 与 CDM 默认 uniform 不读取 upper_bound，
虽然调用处都传了这个参数。CPU 测试确认 upper_bound=0.3 时，uniform 仍有约 69.1% 样本超过 0.3，而 capped 不超过它。</p>
<p>正文/算法的独立 uniform 简写不完整；附录 C.1 提到沿用 D-DMD 的 CA 时间策略，
<a href="https://github.com/byliutao/CDM/issues/3#issuecomment-4419073012">作者在 issue #3 明确确认</a>。
因此不应将 CA 的 capped 采样判为偷偷改动方法。</p>
<h3>fake ← student EMA：有意设计，但论文复现说明缺项</h3>
<p>每次 fake 梯度更新后执行 <code>ψ ← 0.999ψ + 0.001θ</code>。这里混入的是当前 student 参数，
并非从 student 的 EMA shadow 读取。{src("scripts/train.py",694)}、{src("cdm/training/model_utils.py",200)}。</p>
<p>论文算法没有列这一步；<a href="https://github.com/byliutao/CDM/issues/4#issuecomment-4420391021">作者称这是借鉴既有工作以加快 fake 跟踪 student</a>。
所以属于需要复现时保留并披露的额外训练技巧，不能称为 accidental bug。</p>
<h3>外推方向是缓存的 stop-gradient 速度</h3>
<p>rollout 在 <code>torch.no_grad()</code> 下完成，CDM 直接使用缓存 <code>all_v</code>。
新外推点上的 student forward 有梯度，但没有沿外推输入回传至原锚点速度的额外链式项。
{src("scripts/train.py",599)}、{src("scripts/train.py",904)}。</p>
<p>这是一种半梯度实现。若将论文式(7)/(9)直接按全部 θ 依赖做总导数，则多出一条路径；论文算法又把 rollout 标为 no-gradient，
因此这里有停梯度约定的歧义，应确认作者意图，<b>不把它武断判为梯度断错</b>。</p>
<h3>归一化与常数</h3>
<p>代码分母为逐样本 mean-absolute-error，加 1e−8 后下限 clamp=0.1；MSE 没有论文形式中的 1/2。
按相同 mean reduction 比较，梯度整体为 half-MSE 的 2 倍，方向一致。三个 loss 同样处理，属于尺度约定；
在 AdamW 和梯度裁剪下不能简单等同于“学习率乘二”。</p>
<h2>6. 运行与效率问题</h2>
<ul>
<li>README 使用 <code>config/requirements_train.txt</code>，实际文件是 <code>requirements_trian.txt</code>，原命令会找不到文件。{src("README.md",89)}</li>
<li><code>random.Random((epoch,batch_idx))</code> 在 Python 3.11+ 抛 TypeError；本地 Python 3.12 已复现。
README 推荐 3.10，这属于跨版本兼容问题，不是证明推荐环境也无法运行。
{src("scripts/train.py",610)}；<a href="https://docs.python.org/3/library/random.html#random.seed">Python 官方变更说明</a>。</li>
<li>共享锚点已开启，但 CA 与 DM 仍各做一次相同输入的 student forward，并同时保留计算图。
默认无随机 dropout 等差异的前提下，可复用一次预测来省一次有梯度前向；改动时需验证梯度与 checkpointing/FSDP 行为。{src("scripts/train.py",757)}、{src("scripts/train.py",817)}</li>
<li>完整 rollout 平均 14.5 个时间步（N∈1..28），之后才采一个锚点；这是代码确实存在的训练开销，
不能用四步推理成本估计每个训练 iteration。</li>
</ul>
<h2>7. 如果用于后续 SD3-Medium 对照</h2>
<p>建议把这个仓库视为<strong>有核心实现、但需要固化实验协议的研究代码</strong>。
首先修复/隔离评测提示词、统一 FID 真实图与生成图预处理，再冻结 student 更新次数、global batch、动态时间范围、weight decay 和 fake EMA 方案。
SD3 默认全参数路径可作为机制参考；LongCat 必须先明确采用论文 LoRA 还是当前全参数配置。</p>
<p>对你的纹理问题，值得借鉴的是“在外推点的新 student 预测上继续做分布匹配”，而不是把其现成 FID 数字直接加入当前 SDXL 表。
另外论文附录 C.2 明确其 SD3 DMD2 baseline 是作者重实现且不含 GAN，不能将文中相对 DMD2 的收益解释为战胜了完整官方 DMD2 GAN 配方。</p>
<p><b>本次没有修改官方源码，也没有启动 CDM 训练。</b>审查脚本：
<a href="audit_repository.py">配置与数据审查</a>、<a href="audit_cpu.py">原函数 CPU 数值审查</a>。
运行 CPU 脚本需要现有 PyTorch、NumPy、Pillow、clean-fid；不需要模型权重。</p>
</main></body></html>""")
(OUT/"report.html").write_text("".join(parts),encoding="utf8")
print(OUT/"report.html")
