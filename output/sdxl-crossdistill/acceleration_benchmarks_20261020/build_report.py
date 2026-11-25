from pathlib import Path
from html import escape
from collections import Counter
import json

ROOT = Path(__file__).resolve().parent
# 只计正文实验或作者评测说明中已核实的评测用法；引用相关工作不计。
papers = [
("DMD2", "2024", "2405.14867", ["COCO","Parti"], "SDXL 为 COCO-2014 固定10K；SD1.5 为30K；Parti 128题子集用于多样性等。"),
("Few-Step SiD", "2025", "2505.12674", ["COCO"], "SDXL 沿用 DMD2 10K；FID、中心裁剪FID、CLIP。"),
("TDM", "2025", "2503.06674", ["HPS"], "HPS四风格题集；HPSv2.1、Aesthetic、CLIP；另有定制老师分布评测。"),
("ADM / DMDX", "2025", "2507.18569", ["COCO","Parti"], "COCO-2014 10K；Parti多样性；偏好评分。"),
("SenseFlow", "2025–2026", "2506.00523", ["COCO","GenEval","CompBench"], "COCO-2017 5K；FID-T参考老师；GenEval与T2ICompBench。"),
("RMD", "2026", "2603.06136", ["HPS"], "HPS v2.1题集；跨分辨率2+2步，另报实际延迟。"),
("rCM", "2025–2026", "2510.08431", ["GenEval"], "图像部分：Cosmos-Predict2，GenEval；视频部分另列。"),
("DMAD", "2026-10", "https://raw.githubusercontent.com/Yzmblog/DMAD/main/train/image/README.md", ["COCO"], "图像SDXL：COCO-2014固定10K；另有ImageNet-64和视频。"),
("SANA-Sprint", "2025", "2503.09641", ["MJHQ","GenEval"], "正文主协议：MJHQ-30K FID/CLIP、GenEval；不因工具箱支持DPG就计入DPG。"),
("TBCM", "2025–2026", "2511.20410", ["MJHQ"], "MJHQ-30K；SANA 0.6B老师，一步生成。"),
("TwinFlow", "2025–2026", "2512.05150", ["GenEval","DPG","WISE"], "SANA及Qwen-Image；GenEval原始/LLM改写分别报告；WISE原版。"),
("IB-Flow", "2026-07", "2607.09133", ["GenEval","DPG","OneIG"], "2 NFE；FLUX、OpenUni、Qwen；OneIG五个分项。"),
("SwD", "2025–2026", "2503.16397", ["COCO","MJHQ","GenEval"], "COCO-30K与MJHQ-30K；其HPSv3在这些题集上评分，不计HPS官方题集。"),
("π-Flow", "2025–2026", "2510.14974", ["COCO","HPS","OneIG"], "COCO10K、HPS3200、OneIG-EN1120；另有ImageNet-256。"),
("ArcFlow", "2026-02", "2602.09014", ["GenEval","DPG","OneIG","Align5000"], "独立GenEval/DPG/OneIG；Align5000=HPS3200+COCO1800，单独计，不冒充完整COCO。"),
("LogCD", "2026", "https://openaccess.thecvf.com/content/CVPR2026/papers/Xie_LogCD_Local-to-global_Consistency_Distillation_for_Few-step_Image_Generation_CVPR_2026_paper.pdf", ["COCO","MJHQ","GenEval"], "COCO-2017 5K、MJHQ-5K、GenEval。"),
("Stabilizing, Scaling & Enhancing MeanFlow", "2026-05", "2605.17834", ["GenEval"], "FLUX的公开定量GenEval；Hunyuan大模型展示另列，不计为新公共benchmark。"),
("EMF", "2026", "2604.18168", ["GenEval","DPG","HPS"], "文本条件MeanFlow；GenEval、DPG及HPS四风格评分。属于原生少步适配，不是传统老师蒸馏。"),
("REST", "2026-08/09", "2608.09226", ["GenEval","DrawBench","OCR","Pick-a-Pic"], "RL任务留出集评GenEval/OCR/PickScore；域外DrawBench；奖励优化成绩单列。"),
("Hierarchical Distillation (HD)", "2025–2026", "2511.08930", ["MJHQ"], "MJHQ-30K文生图；另有ImageNet-256类别条件实验。"),
("DMDR", "2025–2026", "2511.13649", ["ShareGPT","GenEval","DPG"], "ShareGPT-4o-Image留出提示词做偏好评测，另有GenEval与DPG。不是HPS官方题集。"),
("SD3.5-Flash", "2025–2026", "2509.21318", ["COCO","GenEval","Parti"], "COCO30K、GenEval；507条专家题+Parti子集做人工评测。"),
("RTDMD", "2026-05", "2605.26108", ["DrawBench","GenEval","GenEval2","OCR","HPSv3-task"], "DrawBench质量/偏好，GenEval2官方设置，另按所引用RL协议评GenEval/OCR/HPSv3。"),
("MeanFlowNFT", "2026-07", "2607.15273", ["DrawBench","GenEval2","OCR","HPSv3-task"], "DrawBench、GenEval2官方设置、RL协议OCR/HPSv3；另有视频。"),
]
def link(url, label="原文"):
    if not url.startswith("http"): url="https://arxiv.org/html/"+url
    return f'<a href="{escape(url,quote=True)}">{escape(label)}</a>'
def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>'+''.join(f'<th>{x}</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{x}</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table></div>'

counts=Counter(t for p in papers for t in p[3])
names={"COCO":"MS-COCO（5K／10K／30K）","MJHQ":"MJHQ（30K／5K）","GenEval":"GenEval","DPG":"DPG-Bench","HPS":"HPS 官方3200题集","OneIG":"OneIG-Bench","GenEval2":"GenEval2","CompBench":"T2I-CompBench","Parti":"PartiPrompts（含子集/混合）","WISE":"WISE","DrawBench":"DrawBench","OCR":"OCR任务测试集","Pick-a-Pic":"Pick-a-Pic测试集","ShareGPT":"ShareGPT-4o-Image留出集","Align5000":"Align5000混合题集","HPSv3-task":"HPSv3任务评测"}
order=["GenEval","COCO","MJHQ","DPG","HPS","OneIG","DrawBench","Parti","GenEval2","CompBench","WISE","OCR","Pick-a-Pic","ShareGPT","Align5000","HPSv3-task"]
freqrows=[]
for key in order:
    users=[p[0] for p in papers if key in p[3]]
    freqrows.append([names[key],f'<b>{counts[key]} / {len(papers)}</b>',"、".join(users)])
paperrows=[[str(i),link(p[2],p[0]),p[1],"、".join(names[t] for t in p[3]),p[4]] for i,p in enumerate(papers,1)]

scores=[
["COCO-2014 10K / SDXL / 4步","SiD₂ᵃ Zero-CFG","FID <b>13.25↓</b>；CLIP 0.335","本次核对到的该设置最低FID候选。Anti-CFG是另一运行：FID13.79 / CLIP0.342，不能择优拼接。",link("2505.12674","表2")],
["同协议的近期强对照","DMAD / DMD2","4步 FID <b>14.47 / 19.32</b>","2026年DMAD并非因此自动成为该设置FID第一；发布时间不能代替数值。",link("https://raw.githubusercontent.com/Yzmblog/DMAD/main/train/image/README.md","DMAD")+ " · "+link("2405.14867","DMD2")],
["MJHQ-30K / SANA 0.6B / 1步","TBCM","FID <b>6.52↓</b>；CLIP28.08","已核实的一步扩散蒸馏强结果、其论文比较表第一；未证实为所有骨干全局第一。不得与MJHQ-5K混排。",link("2511.20410","表2")],
["HPS3200 / SDXL / 4步","TDM","HPSv2.1 <b>34.88↑</b>","TDM原论文表内最佳。RMD中同名TDM为31.93，跨论文设置未对齐，不能认证统一全局第一。",link("2503.06674","表1")+ " · "+link("2603.06136","RMD")],
["GenEval / Qwen-Image / 1–2 NFE","TwinFlow（全参、longer training）","1 NFE <b>0.89</b>；2 NFE <b>0.90</b>","未经表内LLM改写标记的结果。其LoRA表另报改写后1/2 NFE为0.90/0.91。不同设置分别记录。",link("2512.05150","表2/3")],
["GenEval / 原生少步文本适配 / 4步","EMF 3B","<b>0.90↑</b>","CVPR2026论文同表高分；不是从0.83的rCM表就可推断整个少步领域最高。",link("2604.18168","表3")],
["GenEval / RL+蒸馏 / 4 NFE","RTDMD，FLUX.2 4B","<b>0.9046↑</b>","该论文RL评测协议；不能自动与553×4、不同提示增强的分数混排。",link("2605.26108","表2")],
["GenEval任务留出 / RL+蒸馏 / 8步","REST-RAM†，SD3.5M","<b>0.94↑</b>","†为GenEval+PickScore奖励优化；是任务留出协议，不直接认证标准GenEval四步SOTA。",link("2608.09226","表1及4.1节")],
["DPG-Bench / Qwen-Image20B / 2 NFE","IB-Flow","<b>88.67↑</b>","本次核对到的两步强结果；同表ArcFlow87.94。论文表2写88.68、表1/正文写88.67，此处采用主表。",link("2607.09133","表1/2")],
["OneIG / Qwen / 2 NFE","IB-Flow 与 Qwen-Image-Lightning各有胜项","IB-Flow：对齐0.877、文字0.863、多样性0.180、风格0.425、推理0.301","IB-Flow表内多样性/风格/推理较强，Lightning对齐0.875、文字0.879。五维不是单一总榜，不能只挑部分称全面SOTA。",link("2607.09133","表1")],
["GenEval2 / 4 NFE / 近期RL蒸馏对照表","TDM-R1，Z-Image6B","<b>0.4073↑</b>","RTDMD表2已核实的最佳条目；RTDMD-FLUX.2为0.2755。尚未建立覆盖全领域、锁定AM/GM及采样协议的排名。",link("2605.26108","表2")],
["DrawBench / 偏好 / SD3.5M / 4步","REST-RAM","PickScore <b>23.97↑</b>","该文同表：RTDMD复现23.71，官方权重23.30。不同骨干/训练数据的23.9642不宜凭0.0058差距判断胜负。",link("2608.09226","表2")],
["CIFAR-10 32² / 1 NFE","Uni-Instruct","无条件 FID <b>1.46↓</b>；类别条件 <b>1.38↓</b>","NeurIPS2025作者报告记录；两种条件分赛道。本次未找到核实后更低的一步对应结果。",link("2505.20755","论文")],
["ImageNet-64 类别条件 / 1 NFE","Uni-Instruct（FKL，longer training）","FID <b>1.02↓</b>","低于DMAD+projected D的1.04；因此1.04不能不加限定地称当前最低。",link("2505.20755","论文")+ " · "+link("https://yzmblog.github.io/projects/DMAD/","DMAD")],
["ImageNet-256 类别条件 / 1 NFE / 允许直接FD后训练","iMF-XL / JiT-H / pMF-H + FD(Inception)","FID <b>0.72↓</b>","直接优化Inception分布统计；需另报FDr⁶或独立表征指标。数字纪录不等于无条件视觉质量第一。",link("2604.28190","附录完整表")],
["ImageNet-256 / 1 NFE / 原生一步、无Fréchet损失","W-Flow XL/2","FID <b>1.29↓</b>","作者公开代码/权重中的记录；从头训练的一步方法，与已有老师蒸馏训练预算不同。",link("https://github.com/hanjq17/W-Flow","官方")],
["ImageNet-256 / 少步蒸馏后精炼 / 无Fréchet损失","FiST augmented","2阶段 FID <b>1.15</b>；3阶段 <b>1.11</b>","2026-09预印本比较表前沿。阶段计算约等于一次SiT评估；3阶段不应写成一步。",link("2609.32376","表1/2")],
]

style="""body{font-family:system-ui,'Microsoft YaHei',sans-serif;background:#f5f7fa;color:#203247;line-height:1.75;margin:0}main{max-width:1460px;margin:auto;padding:30px 28px 70px}header{background:#17384d;color:white;padding:25px 30px;border-radius:12px}h1{font-size:30px;margin:6px 0}h2{font-size:23px;margin-top:34px}a{color:#1767a1}header p{color:#c7dce7}.note{background:#e6f0f7;padding:16px 20px;border-left:4px solid #4388b1;margin:18px 0}.warn{background:#fff3da;border-color:#c69b4b}.muted{color:#68788a;font-size:14px}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px;margin:14px 0}th,td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #dce4ed}th{background:#e6eef5}td:first-child{min-width:130px}li{margin:7px 0}.meter{display:flex;gap:12px;align-items:center;margin:8px 0}.meter span:first-child{width:220px;flex-shrink:0}.track{background:#dfe8f0;width:300px;height:16px}.fill{background:#4586aa;height:16px}nav{display:flex;gap:20px;flex-wrap:wrap;margin:18px 0}code{background:#e7edf3;padding:2px 4px}@media(max-width:700px){main{padding:14px}.meter span:first-child{width:145px}.track{width:130px}h1{font-size:25px}}"""
bars=''.join(f'<div class="meter"><span>{names[k]}</span><div class="track"><div class="fill" style="width:{100*counts[k]/len(papers):.2f}%"></div></div><b>{counts[k]}/{len(papers)}</b></div>' for k in order[:9])
html=f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>扩散推理加速：常用Benchmark与SOTA记录</title><style>{style}</style></head><body><main>
<header><h1>扩散推理加速：常用 Benchmark 与 SOTA 记录</h1><p>范围已纠正：统计这个研究领域的实际评测习惯，不是罗列所有文生图benchmark，也不是部署评测计划。核对日期：2026-10-20。</p></header>
<nav><a href="#usage">使用统计</a><a href="#sota">前沿成绩</a><a href="#papers">逐篇证据</a><a href="#other">机制实验/视频/免训练加速</a><a href="#limits">结论边界</a></nav>
<div class="note"><b>领域主线：</b>文本到图像的少步/蒸馏加速，反复出现的是 <b>COCO、MJHQ、GenEval、DPG、HPS提示词集</b>；新一批方法增加 <b>OneIG、GenEval2、DrawBench及OCR</b>。研究一般生成机制时，<b>CIFAR-10、ImageNet-64、ImageNet-256</b>仍是独立主线。PartiPrompts、T2ICompBench和人类盲评常作补充。</div>
<p><b>调查方法：</b>人工核对下面24篇有文本到图像实验的加速/原生少步/少步RL论文，记录其实际报告的评测集。只在相关工作中出现不计；使用HPS评分器不自动计作HPS官方题集；混合题集单列。下表为这个明确样本中的<b>已确认出现次数下界</b>，不是数据库穷尽统计，也不是领域论文占比。补充的类别条件、视频和缓存论文不放进该分母。</p>
<h2 id="usage">1．这个领域通常用哪些 Benchmark</h2>
{bars}
{table(["评测集/协议","已确认出现","实际使用论文"],freqrows)}
<p><b>规模口径：</b>COCO常见2014固定10K/30K与2017 val5K；MJHQ通常30K，也有5K子集；GenEval通常553题×4图；DPG为1065题，官方推荐每题4图；HPS标准题集3200题；OneIG-EN1120题；GenEval2为800题；DrawBench200题；T2ICompBench6000题。相同名称下的子集、裁判和采样次数仍须看原文。</p>
<p>在我们的24篇样本中，<b>HPS评分器的使用远多于HPS题集</b>。例如SwD在COCO/MJHQ图片上算HPSv3，DMDR在ShareGPT留出提示上算HPS；这些成绩不属于HPS3200官方题集的排行榜。OCR也必须注明具体任务集，不能见到“OCR”就认定是LongText或CVTG-2K。</p>
<h2 id="sota">2．各 Benchmark 的 SOTA / 已核实前沿记录</h2>
<div class="note warn"><b>怎么读“SOTA”：</b>本领域没有覆盖全部骨干、步数和协议的统一官方排行榜。下表给出本次核对到的领先记录，并明确区分：对应设置的最低/最高候选、作者同表最佳、尚不能认证全局第一。不同条件不强行排成一列。全表都是公开报告值，不是本次重测。</div>
{table(["Benchmark及赛道","方法/骨干","公开成绩","SOTA证据与限制","来源"],scores)}
<h3>为什么没有给每个名字硬填一个冠军</h3>
<p><b>PartiPrompts</b>常用于自选子集的人类偏好和多样性，比较对象不同就没有统一胜率冠军；<b>T2ICompBench</b>是多类别得分，需明确取哪个类别/是否取平均；<b>OneIG</b>在相关加速论文中常按五维分别报，胜者可以不同；<b>FID-T / Align5000</b>的参考分布由老师决定，不能跨老师认定全局SOTA。COCO-5K/30K的协议差异也不足以支持本次直接认证全局最低值。</p>
<p><b>尤其不能把rCM的GenEval 0.83称为当前所有少步图像模型的SOTA：</b>它是Cosmos14B四步的具体设置；EMF、TwinFlow以及奖励优化后的少步方法已有不同设置下更高的报告值。方法进步与老师能力、文本编码器、提示改写和奖励训练需要一并记录。</p>
<h2 id="papers">3．24篇文本到图像论文逐篇证据</h2>
{table(["序号","论文/作者说明","时间","确认用于评测的集合","协议备注"],paperrows)}
<h2 id="other">4．补全领域另外几条评测主线</h2>
<h3>类别条件/无条件生成：常用于方法与理论验证</h3>
{table(["Benchmark","领域用途","代表论文/记录"],[
["CIFAR-10 32²","无条件/类别条件分开；常见50K生成样本算FID，辅以IS。",link("2505.20755","Uni-Instruct")+"；CTM、SiD等传统蒸馏主线。"],
["ImageNet-64","类别条件；DMD/SiD类分布匹配常用。",link("2405.14867","DMD2")+"、"+link("2505.20755","Uni-Instruct")+"、"+link("https://yzmblog.github.io/projects/DMAD/","DMAD")],
["ImageNet-256（及512扩展）","DiT、MeanFlow、consistency、flow-map等核心试验；FID50K、IS、Precision/Recall，部分报sFID/FDr。",link("2512.02012","iMF")+"、"+link("2602.10764","DE-CM")+"、"+link("2606.15553","Drifting distillation")+"、"+link("2609.32376","FiST")],
["FFHQ-64、AFHQv2-64 / LSUN等","单域生成的辅助检验；不同分辨率和类别分别比。",link("2405.20750","f-distill")+"报告FFHQ64一步FID0.85、AFHQv2-64一步1.23；属于已核实历史强结果，本次未认证其2026全局第一。"],
])}
<h3>视频推理加速：独立赛道</h3>
{table(["基准","相关论文与用法","公开前沿参照及限制"],[
["VBench","CrossDistill、rCM、DMAD、MeanFlowNFT等检查视频质量和语义；另报吞吐/首帧延迟/多样性。",link("2609.14725","CrossDistill")+"：Wan1.3B四步83.77、14B四步84.46；"+link("https://yzmblog.github.io/projects/DMAD/","DMAD")+"：Wan14B四步85.15。后者是更高公开值，不足以认证所有视频骨干的统一四步冠军。"],
["VBench2","更难的视频能力评测；SwD明确使用。",link("2503.16397","SwD")+"表2：自家Wan1.3B设置四步53.2；这是同表领先，不是全领域SOTA。"],
["视频多样性/人类偏好","同提示多种子：V-JEPA2、VideoMAEv2、奖励模型等。","CrossDistill使用5种子；人类胜率和嵌入多样性依赖比较对象，不能抽出一个无条件总榜。"],
])}
<h3>缓存、剪枝与免训练采样加速</h3>
<p>这条路线常复用COCO、PartiPrompts、ImageNet等，重点是<b>相对原模型的质量保持＋实测加速比</b>；PSNR、SSIM、LPIPS可能用于相同种子输出的一致性。
例如 {link("https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_DeepCache_Accelerating_Diffusion_Models_for_Free_CVPR_2024_paper.pdf","DeepCache")}使用COCO2017与PartiPrompts，
{link("https://openreview.net/pdf/4ccc37468630e5de606b67598074bc7bdc6f590a.pdf","EcoDiff")}用COCO与Flickr30K。这里的SOTA应是规定质量损失下的速度/显存前沿，不能拿另一个更强老师的绝对FID作为缓存算法冠军。</p>
<h2 id="limits">5．本次调研能得出的结论</h2>
<ul>
<li><b>最常见的并非几十个泛化能力榜单。</b>图像推理加速的主体仍围绕COCO/MJHQ分布指标、GenEval/DPG语义、HPS/DrawBench偏好展开。</li>
<li><b>2026年的扩展已实际出现。</b>OneIG有π-Flow、ArcFlow、IB-Flow；GenEval2有RTDMD、MeanFlowNFT；这些是领域采用证据，不是因为benchmark新就推荐全跑。</li>
<li><b>现代少步方法的基准已经超出SDXL。</b>比较应保留每个结果的骨干、1/2/4/8 NFE、CFG、提示改写和奖励训练标签。</li>
<li><b>最小FID也必须看训练目标。</b>ImageNet256一步0.72用了直接FD优化，不能与不使用Fréchet损失的1.29/三阶段1.11忽略差异地比较研究贡献。</li>
</ul>
<p class="muted">未启动训练、下载权重或运行benchmark。本报告替代上一份“通用文生图全景”作为当前问题的回答；原报告仅保留作其他任务的背景资料。可机读的逐篇统计见同目录 <a href="paper_inventory.json">paper_inventory.json</a>。</p>
</main></body></html>"""
ROOT.mkdir(parents=True,exist_ok=True)
(ROOT/"report.html").write_text(html,encoding="utf-8")
(ROOT/"paper_inventory.json").write_text(json.dumps({"as_of":"2026-10-20","scope":"24 selected T2I acceleration and native few-step papers; verified usage lower bounds, not exhaustive bibliography","papers":[dict(name=p[0],year=p[1],source=p[2] if p[2].startswith("http") else "https://arxiv.org/html/"+p[2],benchmarks=p[3],notes=p[4]) for p in papers],"counts":dict(counts)},ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps({"papers":len(papers),"counts":dict(counts),"report":str(ROOT/"report.html")},ensure_ascii=False))
