"""Build a standalone literature review from already archived project results.

No training, remote access, queue changes, or modifications to experiment sources.
"""
from pathlib import Path
import json
import hashlib
import html
from html.parser import HTMLParser

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT / 'output/sdxl-crossdistill'
old_path = BASE / 'fid_pfid_comparison_20261113_side/results_and_recipes.json'
new_path = BASE / 'new_models_eval_20261114/summary_four_complete.json'
old = {x['key']: x for x in json.loads(old_path.read_text(encoding='utf-8'))}
new = json.loads(new_path.read_text(encoding='utf-8'))
terminal = new['A_terminal_only']['details']
rows = [
    ('A：PCM＋中段DM＋末步DM', old['A']),
    ('A：去中段学生DM，fake保留中段', terminal),
    ('C＋1%：PCM＋中段DM＋末步DM', old['C_mix1']),
    ('C＋1%：去中段学生DM，fake保留中段', old['PCM_DM4']),
]

def table(data):
    return '<table><thead><tr><th>G160 实验</th><th>FID ↓ / 10K</th><th>Patch FID ↓ / 10K</th><th>CLIP ↑ / 1024</th></tr></thead><tbody>' + ''.join(
        f'<tr><td>{html.escape(name)}</td><td>{d["FID"]:.4f}</td><td>{d["Patch_FID"]:.4f}</td><td>{d["CLIP"]:.4f}</td></tr>' for name,d in data
    ) + '</tbody></table>'

# Static figure: shared zero-based scale; all values are read from archived JSON.
svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="340" viewBox="0 0 1100 340" role="img" aria-label="两组去中段学生DM前后FID与Patch FID">',
       '<rect width="1100" height="340" fill="#fff"/>',
       '<style>text{font-family:Arial,Microsoft YaHei,sans-serif;fill:#223349;font-size:15px}</style>']
for j,metric in enumerate(('FID','Patch_FID')):
    left=250+j*415
    svg.append(f'<text x="{left}" y="28" font-weight="bold">{metric} ↓</text>')
    for tick in range(0,41,10):
        x=left+tick*7
        svg.append(f'<line x1="{x}" x2="{x}" y1="43" y2="290" stroke="#e5eaf0"/><text x="{x}" y="315" text-anchor="middle">{tick}</text>')
    for i,(label,d) in enumerate(rows):
        y=57+i*58
        color='#0d8c7a' if i%2 else '#8295b2'
        svg.append(f'<rect x="{left}" y="{y}" width="{d[metric]*7:.2f}" height="28" fill="{color}" rx="3"/><text x="{left+d[metric]*7+8:.2f}" y="{y+20}">{d[metric]:.2f}</text>')
for i,label in enumerate(('A · 有中段学生DM','A · 无中段学生DM','C＋1% · 有中段学生DM','C＋1% · 无中段学生DM')):
    svg.append(f'<text x="20" y="{77+i*58}">{label}</text>')
svg.append('</svg>')
(OUT/'comparison.svg').write_text('\n'.join(svg),encoding='utf-8')

body = '''
<header><div class="eyebrow">SDXL · 文献与实现核对 · 2026-11-15</div>
<h1>为什么去掉中段 DM，当前结果反而更好？</h1>
<p class="lead">现有证据支持：在当前 G160 预算和实现下，中段学生 DM 的净收益为负。更值得怀疑的是监督信号的估计质量、梯度分配与区间设计；还不能据此认定“中间分布监督没有价值”。</p></header>

<section><h2>1．先确定我们究竟观察到了什么</h2>
<p>这里的“去中段”只删除<strong>学生</strong>第2/3步的DM损失。首步PCM、末步DM仍在；fake仍做中段一次、末步一次更新。四步共用一套学生UNet，所以没有单独的中段损失，不等于中段网络参数被冻结。</p>
__MAIN_TABLE__
<img class="chart" src="comparison.svg" alt="去中段学生DM前后的FID与Patch FID对比"/>
<p>A去中段后，FID下降<strong>3.1800</strong>，Patch FID下降<strong>2.7465</strong>；C＋1%对应下降<strong>3.5277 / 4.9430</strong>。两个配方下方向一致，值得认真解释。CLIP分别下降0.0661、0.1356，尚无多训练种子或置信区间，因此应表述为<strong>分布指标改善，语义分数略降</strong>，而不是所有质量都提升。</p>
<div class="note">数据口径：项目归档的固定COCO-2014 10K评测；Patch FID为中心299×299裁剪；CLIP为1024样本。G160是160次学生更新，不是160轮数据集。此处没有重跑评测，也没有读取实时服务器状态。</div>
<p><strong>两个重要排除：</strong>同为G160，去中段并没有因此多做PCM更新；fake更新也没减少，不能把收益解释为“省出的fake算力全部给了末步”。删掉的是学生接收的一股梯度，以及由此引起的后续优化轨迹变化。</p>
</section>

<section><h2>2．论文与开源实现究竟怎么说</h2>
<h3>DMD2：fake跟踪质量重要，但它不是我们的“只训第4步”</h3>
<p>DMD2将去掉回归约束后的不稳定归因于fake对生成分布估计不充分，并用更频繁的fake更新缓解。其官方多步代码每次随机选择一个生成位置，无梯度推进前缀，在选中位置预测clean x₀再做DM；不是每次只监督推理第4步，也不是同时叠加中段与末步两项DM。<a href="https://arxiv.org/html/2405.14867v2">论文§4.2–4.5</a>；<a href="https://github.com/tianweiy/DMD2/blob/main/main/sd_unified_model.py">官方sample_backward与forward实现</a>。</p>
<p class="interpret">对本项目的启发：不能用“DMD2也只训末步”证明我们正确；可以借鉴的是，先确认fake能提供可靠方向，再增加学生监督项。</p>

<h3>TDM：中间带噪分布可以有效；区间和fake的分工有专门设计</h3>
<p>TDM论文直接匹配中间带噪状态，特意用不重叠的加噪区间，让时间条件区分各阶段分布；反传只穿过一个ODE步。论文还报告了带噪目标优于中间clean预测的消融。这是“中间DM必然无益”的反例。<a href="https://arxiv.org/html/2503.06674v2">TDM §3.1、附录F.2</a>。</p>
<p>但不能只读论文早期表述：<strong>官方README也支持从阶段边界一直加噪到T的宽区间，并建议给fake加阶段条件</strong>。因此我们放宽上界、加入阶段嵌入，有官方设计依据；“区间重叠”本身不是已证实错误。<a href="https://github.com/Luo-Yihong/TDM#usage">官方仓库训练说明</a>。</p>

<h3>VA-TDM：最贴近这次问题的新证据</h3>
<p>2026年11月4日的VA-TDM区分了两个问题：①从实际中间带噪状态继续扰动，直接DSM目标可以总体正确，但方差较高；②改用clean目标训练，可能估计的并非学生实际路径的扰动分布。它采用带阶段条件的在线场及双形式监督处理这些困难。<a href="https://arxiv.org/html/2611.06796v1">VA-TDM §3.1、附录B</a>；<a href="https://github.com/Luo-Yihong/VA-TDM">作者代码</a>。</p>
<p class="interpret">与我们的对应：当前boundary.py已经从真实生成的中间latent加噪，并使用相应条件score目标，不能直接给它扣上“clean目标错配”的结论。更贴近的是第一类：<strong>正确的总体目标，也可能因为估计方差和在线追踪难度而在短训练中不好用</strong>。论文的参考过程分析也不是对本项目SDXL误差大小的实测证明。</p>

<h3>D-DMD：增加一项带CFG的DM，也增加了语义增强作用</h3>
<p>D-DMD把通常合在一起的更新拆成条件分布匹配与CFG增强，并分别研究加噪日程；这提醒我们，“增加中段DM”并非只增加一个中性的匹配约束。<a href="https://arxiv.org/html/2511.22677v1">D-DMD §3–4</a>。</p>
<pre>teacher_CFG − fake_cond
= (teacher_cond − fake_cond)
  + (CFG − 1) × (teacher_cond − teacher_uncond)</pre>
<p class="interpret">我们的教师CFG为8。中段DM还会对中间表征施加这一额外语义方向。去中段后CLIP略降而FID/Patch改善，与语义压力和局部分布之间的权衡相容，但不能仅凭这一相关性确认原因。</p>
</section>

<section><h2>3．GitHub讨论中有哪些可用的经验</h2>
<table><thead><tr><th>来源与身份</th><th>具体讨论</th><th>本项目应如何使用</th></tr></thead><tbody>
<tr><td><a href="https://github.com/Luo-Yihong/TDM/issues/8">TDM #8</a><br>复现者＋作者Luo-Yihong</td><td>复现者迁移SD3时出现噪点和模糊；改为v预测后缓解。作者指出x₀与v都可以工作，但x₀回归应配合SNR权重。</td><td>代数等价不等于在当前加权和有限训练下优化等价。我们的条件目标权重需要单独诊断，不能只看目标名称。</td></tr>
<tr><td><a href="https://github.com/Luo-Yihong/TDM/issues/9">TDM #9</a><br>作者确认并修复</td><td>公开代码混合噪声公式漏了一个σ系数；作者确认错误并说明已修复。</td><td>论文代码也需核算。我们当前是a·xₛ＋b·ε，不能因为这个issue就声称也有同一bug。</td></tr>
<tr><td><a href="https://github.com/Luo-Yihong/TDM/issues/13">TDM #13</a><br>作者经验</td><td>作者称去掉归一化权重时，Huber仍能工作，而MSE性能明显变差。</td><td>我们用归一化方向＋MSE；应看归一化后的梯度、分位数与裁剪，不能只比原始loss数值。不是要求立即换Huber。</td></tr>
<tr><td><a href="https://github.com/Luo-Yihong/TDM/issues/3">TDM #3</a><br>作者解释</td><td>作者说明fake监督对应生成路径上的clean预测，并在每次迭代抽取区间内的时间。</td><td>公式的阶段求和，不自动等价于每次把所有阶段的loss全部反传。</td></tr>
<tr><td><a href="https://github.com/tianweiy/DMD2/issues/73">DMD2 #73</a><br>作者tianweiy</td><td>GAN与DMD权重先参考传到生成图像上的梯度尺度，再做小范围搜索；判别相关项太强可破坏fake。</td><td>支持“看梯度而非loss数量级”。该建议针对GAN/DMD，不是直接给PCM指定一个比例。</td></tr>
<tr><td><a href="https://github.com/Tongyi-MAI/Z-Image/issues/56">Z-Image #56</a><br>复现者＋协作者ChrisLiu6</td><td>讨论质量先升后降、网格伪影；协作者建议检查fake学习率，并解释增大CFG会提高增强项相对匹配项的比例。</td><td>fake跟踪与CFG平衡是真实工程问题。但这是别人的训练反馈，不能充当我们fake跟不上的实测证据。</td></tr>
</tbody></table>
<p>另见<a href="https://github.com/Luo-Yihong/TDM/issues/7">TDM #7</a>：作者把随机输出边界解释为生成器正则化。这说明TDM实现还包含目标边界设计；不能把“用了中段DM”四个字当成完整方法。</p>
</section>

<section><h2>4．回到代码：更值得怀疑的三个机制</h2>
<h3>第一：中段梯度可能挤占或抵消末步与PCM，而不只是“多学一点”</h3>
<pre>原A：g = g_PCM + g_随机中段 + g_末步
去中段A：g = g_PCM + g_末步
阶段2/3均匀选择时：E[g_随机中段] = ½g₂ + ½g₃</pre>
<p>源码按各项反传后累加，不是按损失项数量求平均。去掉中段不会把PCM梯度自动乘1.5，也不会多做PCM更新。它改变的是向量和、裁剪后的方向及Adam的历史状态。“PCM占比提高”可以是其中一种解释，但<strong>末步DM的相对地位也同时提高</strong>。</p>
<p>直观例子：一套共享UNet既要满足PCM的一致性，又要让中间分布靠近老师，还要优化最终图像。如果中段方向与后两者相反，新增监督反而让最终图像更新走偏；即便方向不反，也可能过强而支配更新。当前只是机制假设，尚未拿到本次探针的实际余弦数值。</p>
<p>还应区分局部梯度与全链路目标：当前rollout在阶段间停止梯度，末步损失没有通过完整链条回传到中段输入；它通过共享参数间接改变前面各步。中段匹配“有自己的理由”，但不保证其局部梯度就是最终图像目标想要的方向。</p>

<h3>第二：中段fake的总体目标正确，有限训练的方向仍可能不可靠</h3>
<p>直接从归档代码核对，边界s到加噪时间u的计算为：</p>
<pre>a = αᵤ / αₛ
b² = σᵤ² − a²σₛ²
xᵤ = a · xₛ + b · ε
L_fake = mean[min(1/b², 10) · (b·ε_fake − σᵤ·ε)²]</pre>
<p>若不截断权重，它等价于回归(σᵤ/b)ε；最优条件均值对应被扰动的实际中间分布score。它<strong>不是把xₛ误当干净图直接套普通加噪</strong>。但当u靠近s时，新增噪声b较小，回归标签可能更难估计；截断权重又改变有限容量模型的时间分配。不能把这种难度直接等同于loss爆炸，也不能把总体正确等同于fake已经学准。</p>
<p>学生实际使用的方向还经过后验形式归一化：</p>
<pre>scale = mean_abs(xₛ − teacher_posterior_mean).clamp_min(1e−5)
direction = (b² / a) · (score_fake − score_teacher_CFG) / scale</pre>
<p>这是带停止梯度归一化的代理方向。不同s、u下，fake误差、归一化分母和最终梯度尺度并不相同；相同“DM系数1”不意味着中段和末步贡献相当。值得记录的是分桶的方向RMS、分母分位数、梯度夹角与裁剪，而不是把不同阶段训练loss的高低直接解释成跟踪能力。</p>
<div class="note"><strong>这条解释的关键边界：</strong>去中段组仍训练中段fake，因此改善不表明fake容量负担已减少；它说明学生不再直接使用那部分中段估计可能有好处。要证明是fake误差而非目标冲突，需要额外诊断。</div>

<h3>第三：宽区间中段监督，可能引入收益不高的全局/语义约束</h3>
<p>A的各阶段加噪区间是509–979、259–979、20–979；中段和末步在高噪区域高度重叠。虽然阶段嵌入解决了“同一个时间到底属于哪一阶段”的识别问题，有限容量fake仍需交替拟合多个变化分布；高噪下局部细节信息又被削弱。于是中段损失可能更多改变全局特征，却增加了纹理目标的优化负担。</p>
<p>这并非“高噪监督一定不好”：官方TDM也允许宽区间。这里的怀疑对象是<strong>宽区间＋当前权重/归一化＋在线fake＋G160</strong>的组合，必须靠控制变量判断。</p>
</section>

<section><h2>5．现有结果里，还有两条不能忽略的反证</h2>
<table><thead><tr><th>已有实验</th><th>FID ↓</th><th>Patch FID ↓</th><th>告诉我们什么</th></tr></thead><tbody>
<tr><td>A：PCM×10，仍有中段DM</td><td>24.6412</td><td>35.2295</td><td>增大PCM能改善FID，却没有复现去中段的Patch收益；不能把二者简单视为同一件事。</td></tr>
<tr><td>A：中段clean＋全局20–979</td><td>23.7370</td><td>42.2395</td><td>clean和全局加噪改善部分FID，但纹理分布指标仍差；“改clean就能消除问题”不成立。</td></tr>
<tr><td>早期局部边界：归一化转向</td><td>22.9175</td><td>26.6626</td><td rowspan="2">两组保留中段DM，Patch FID仍优于上面的去中段组；中段DM并非普遍有害。</td></tr>
<tr><td>早期局部边界：原始转角</td><td>23.0569</td><td>25.8994</td></tr>
</tbody></table>
<p>早期局部组只在509–739、259–489、20–239加噪，且先验、fake轮转等也不同；它们<strong>不是</strong>与A的严格配对区间消融，不能把收益全算给局部区间。但它们已经足以反驳“只要加中段DM就差”。PCM×10也不是“去中段后的相对比例”的精确等价实验，10倍可能过大。</p>
<p>因此我不建议现在把中段DM彻底否定。更有价值的问题是：<strong>怎样的中段监督，在什么区间和强度下，能与末步目标相容？</strong></p>
</section>

<section><h2>6．结论强度：还需要扣除哪些混杂</h2>
<ul>
<li><strong>随机流问题确实影响这次归因。</strong>11月14日历史审计证明，多阶段学生的加法种子存在跨阶段、跨更新碰撞；A去中段组正式学生只有末步，已不触发同类跨阶段碰撞。旧C＋1%去中段组学生问题减少，但fake先验流仍有相关性。因此“有中段/无中段”也改变了随机相关结构。真实FID不因此失效，但不能当作修复后算法的无偏因果结论。</li>
<li><strong>不能把复用随机键算成重复图片或少训练多少数据。</strong>提示词、初始生成噪声和模型参数仍在变化；现有审计也不能量化它贡献了多少FID差距。</li>
<li><strong>短预算与单种子。</strong>160×64＝10240次学生样本暴露，对10K规模只是约一遍量级，不等于每条必然恰好见过一次。多项监督可能收敛更慢，也可能长期确实冲突，当前无法区分最终上限。</li>
<li><strong>指标维度不同。</strong>FID不是像素MSE；Patch FID也只反映中心裁剪的特征分布。CLIP略降、纹理是否更自然、是否有伪细节，应与固定样图共同解释。</li>
<li><strong>尚无直接fake跟踪证据。</strong>训练loss混合不同u、样本与不断变化的学生分布；过去部分实验未存fake，不能用学生最终权重补出历史fake留出loss。</li>
</ul>
</section>

<section><h2>7．我建议下一步怎样验证，先后顺序是什么</h2>
<h3>先用已有诊断，暂不增加复杂训练机制</h3>
<p>项目已有固定8样本、G1/40/80/120/160的梯度探针代码。本次找到的是源码，未取得这五个时点的实际记录，所以没有编造“负余弦”结论。优先读取以下量：</p>
<table><thead><tr><th>检查</th><th>结果如何解释</th></tr></thead><tbody>
<tr><td>cos(g₂,g₄)、cos(g₃,g₄)，以及各中段与PCM的夹角</td><td>持续负向且幅度明显，支持共享参数目标冲突；偶尔负向不足以定论。</td></tr>
<tr><td>中段梯度对g_PCM＋g₄的投影；PCM、中段、末步加权范数</td><td>区分“方向相反”和“只是过强”。不要用三个loss的数值比例代替。</td></tr>
<tr><td>按阶段与u分桶的fake误差、方向RMS和归一化分母</td><td>查中段是否在特定区间失真；不同目标应报告归一化误差，不能直接横比原始MSE。</td></tr>
</tbody></table>
<p>探针是在去中段模型的状态上临时计算中段梯度，只反映该模型位置的局部兼容性，不代表原A整条训练轨迹；固定8样本也不是实际batch64梯度。Adam下真实参数更新还受历史二阶矩影响。</p>

<h3>若只愿意再花一组G160：首选中段DM降权，而不是马上增加fake网络</h3>
<p><strong>候选：A的其余设置不变，L＝PCM＋0.25×随机中段DM＋末步DM，fake仍是中段一次＋末步一次。</strong>0.25是建议的诊断强度，不是论文证明的最优值。它能看出是否存在“弱中段约束比全开更合适”的迹象。若只比现有旧A/去中段，只能作为探索性证据；严格结论需统一修复随机流后补0和1两个端点，不能拿新随机流的一组和旧结果做纯单变量宣称。</p>
<p>若要专门检验“PCM比例”而不是中段本身，可另外讨论<strong>2×PCM＋中段DM＋末步DM</strong>，以及<strong>PCM＋中段DM＋2×末步DM</strong>。前者加强轨迹一致性，后者加强最终分布。它们不是严格数学等价，只用于区分哪个方向被挤占；不建议一次全排队。</p>

<h3>若梯度不冲突却中段方向很噪，再查fake</h3>
<p>在未来可取得fake状态的诊断点，冻结学生与输入分布，留一批生成样本不用来拟合，对fake做短暂离线追踪测试；看留出误差和中段DM方向是否稳定、是否反复翻转。若多拟合能稳定方向，才更支持“fake跟不上”，再考虑增加中段fake更新或调整目标权重。这里是诊断建议，不是已经执行的实验。</p>
<p>若fake稳定而冲突仍在，再做局部区间对照：保持A无先验、阶段条件、fake配对和损失权重，只收窄中段的加噪上界；末步20–979保持原样。这样才能把“中段区间”的影响与末步变化分开。不要直接拿早期局部先验组代替这个配对实验。</p>
<p class="decision"><strong>当前推荐：</strong>实际训练基线可以先用PCM＋末步DM；研究上保留中段DM方向。先看梯度与fake方向质量，再决定是降权、收窄中段区间，还是改fake估计。当前没有足够证据支持直接堆更多阶段损失、更多网络或直接认定PCM×10能解决一切。</p>
</section>

<section><h2>8．证据与复核范围</h2>
<p>项目数值来自本地已归档JSON，机制核对来自实际归档源码；论文和GitHub讨论于2026-11-15读取。没有修改训练代码、没有启动或停止实验、没有改变主线程任务队列，也没有连接训练服务器。本报告不是完整训练代码审计。</p>
<ul>
<li><a href="../new_models_eval_20261114/summary_four_complete.json">最新四组完整分数</a></li>
<li><a href="../fid_pfid_comparison_20261113_side/results_and_recipes.json">历史分数与配置汇总</a></li>
<li><a href="../A_terminal_only_20261113/source/boundary.py">边界加噪、fake目标和学生代理方向</a></li>
<li><a href="../A_terminal_only_20261113/source/train.py">学生/fake更新、PCM和梯度累积</a></li>
<li><a href="../A_terminal_only_20261113/source/pcm_middle_probe_n13.py">固定样本梯度探针</a></li>
<li><a href="../rng_scope_audit_20261114/影响范围与结论.txt">历史随机流影响审计</a></li>
<li><a href="evidence.json">本次引用文件SHA-256、数值与来源链接清单</a></li>
</ul>
<p class="muted">标签约定：“实测”仅指归档指标；“代码事实”指本次读取源码；“作者结论”只归属于原论文或作者回复；“推断/建议”不表示本项目已验证。公开仓库main分支会更新，报告中的公式对应本次读取版本。</p>
</section>
'''
body = body.replace('__MAIN_TABLE__',table(rows))
css = '''body{margin:0;background:#f1f4f8;color:#223349;font:16px/1.85 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1120px;margin:30px auto;padding:0 24px}header,section{background:white;border:1px solid #dce4ec;border-radius:12px;padding:30px 36px;margin:20px 0}h1{font-size:32px;line-height:1.4;margin:10px 0}h2{font-size:23px;color:#123f64;margin:0 0 18px}h3{font-size:19px;margin:24px 0 8px}.eyebrow,.muted{color:#60758c}.lead{font-size:19px}.note,.interpret,.decision{border-left:4px solid #188f80;padding:12px 18px;background:#edf7f5}.note{border-color:#cb962d;background:#fff8e7}table{border-collapse:collapse;width:100%;font-size:15px;margin:16px 0}th,td{border-bottom:1px solid #dfe6ee;padding:12px;text-align:left;vertical-align:top}th{background:#edf2f7}a{color:#1769aa;overflow-wrap:anywhere}pre{background:#f2f5fa;padding:18px;overflow:auto;line-height:1.7;font-size:15px}.chart{width:100%;height:auto}li{margin:8px 0}strong{color:#163e60}@media(max-width:650px){header,section{padding:20px 16px}main{padding:0 10px}table{font-size:12px}td,th{padding:7px}h1{font-size:26px}}@media print{body{background:white}main{max-width:none;margin:0;padding:0}header,section{border:0;padding:12px 0;break-inside:auto}h2,h3{break-after:avoid}tr{break-inside:avoid}a{color:#223349}}'''
document = '<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>中段DM为何可能拖累当前训练｜论文与GitHub调研</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>'
(OUT/'report.html').write_text(document,encoding='utf-8')

class Plain(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.link=None
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag in ('p','h1','h2','h3','li','tr','pre','section'): self.parts.append('\n')
        if tag in ('td','th'): self.parts.append(' | ')
        if tag=='a': self.link=attrs.get('href')
    def handle_endtag(self,tag):
        if tag in ('p','h1','h2','h3','li','tr','pre','section'): self.parts.append('\n')
        if tag=='a' and self.link:
            self.parts.append(' ['+self.link+']'); self.link=None
    def handle_data(self,data): self.parts.append(data)

plain=Plain(); plain.feed(body)
(OUT/'调研报告.txt').write_text(''.join(plain.parts),encoding='utf-8')
sources=[old_path,new_path,BASE/'A_terminal_only_20261113/source/boundary.py',BASE/'A_terminal_only_20261113/source/train.py',BASE/'A_terminal_only_20261113/source/pcm_middle_probe_n13.py',BASE/'rng_scope_audit_20261114/影响范围与结论.txt']
links=[]
class Links(HTMLParser):
    def handle_starttag(self,tag,attrs):
        u=dict(attrs).get('href','')
        if tag=='a' and u.startswith('https://') and u not in links: links.append(u)
Links().feed(body)
evidence={'date':'2026-11-15','scope':'Side-conversation literature review; archived local data, no remote experiment actions','main_comparisons':[{'name':n,'FID':d['FID'],'Patch_FID':d['Patch_FID'],'CLIP':d['CLIP'],'fid_n':10000,'clip_n':1024} for n,d in rows], 'files':[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources], 'public_sources':links, 'not_measured':['Actual gradient probe values were not retrieved','No new fake held-out evaluation','No new training or FID calculation']}
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'report':str(OUT/'report.html'),'text':str(OUT/'调研报告.txt'),'sources':len(links),'local_evidence_files':len(sources)},ensure_ascii=False))
