from pathlib import Path
import json,html
root=Path('output/sdxl-crossdistill')
out=root/'lab_results_review_20261102'
scores=json.loads((root/'coco_uniform_g160_20261025/combined_scores.json').read_text())
labels={'dmd_only':'旧四点：纯DMD','gan4':'旧四点：末步纯GAN','joint4':'旧四点：末步GAN+DMD','joint34':'旧四点：第3、4步GAN+DMD','joint4_uniform':'均匀四点：末步GAN+DMD','sdxl':'原始SDXL，50步','dmd2':'官方DMD2，4步'}
rows=''.join(f'<tr><td>{labels[k]}</td><td>{scores[k]["n"]:,}</td><td>{scores[k]["FID"]:.4f}</td><td>{scores[k]["Patch_FID"]:.4f}</td></tr>' for k in labels)
delta=scores['joint4']['Patch_FID']-scores['dmd_only']['Patch_FID']
report=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验室训练成果与下一步讨论</title>
<style>body{{font:16px/1.85 system-ui,"Microsoft YaHei";max-width:1120px;margin:40px auto;padding:0 24px;color:#203044}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #ccd6df;padding:10px;text-align:left}}aside{{padding:16px;background:#edf3f8}}h2{{margin-top:35px}}a{{color:#205ba7}}</style>
<h1>实验室服务器训练成果与下一步实验讨论</h1><p>整理日期：2026-11-02。实验室 .33.60 / .32.81 的训练状态和配置本次重新读取；COCO分数读取已保存的完整评测记录，并未重新跑评测。此次没有启动、停止或修改训练。G160表示160次学生更新，不是160个epoch。</p>
<aside><strong>可确认主线：</strong>5个自训G160模型完成同协议COCO2014 10K评测；旧版局部边界DM归一化先验组完成G160；原始转角组最新快照G105、仍在运行；无先验局部边界基线只到G6。当前租赁A800的B/C另列，避免和实验室旧先验混淆。</aside>
<h2>1. 已完成的质量评测</h2><table><tr><th>模型</th><th>评测图像数</th><th>FID↓</th><th>Patch FID↓</th></tr>{rows}</table>
<p>五个自训模型使用LAION衍生10K缓存，batch64、学生学习率1e-6。旧四点为999/919/499/199；均匀四点为999/749/499/249。旧主线每G训练fake五次，G160共800次fake更新；不是当前边界主线的fake两次配置。四组旧四点中末步纯GAN使用双卡，其余三组最终版使用四卡；评测文件记录了具体配置。</p>
<p>评测使用相同COCO2014 10K提示词、种子和Inception流程。FID预处理为中心正方形、Lanczos512、clean299；Patch FID为原图中心299×299裁剪。该分数是项目内重评，不能直接视为论文排行榜；中心Patch FID也不等于全图纹理准确率。官方DMD2和原SDXL是质量参照，其训练预算不同。</p>
<p><strong>能够支持：</strong>旧四点下末步联合GAN让Patch FID从52.6431降至48.1145，改善{-delta:.4f}（{-delta/scores['dmd_only']['Patch_FID']*100:.2f}%），FID变化很小；属于值得复验的局部分布改善信号。末步纯GAN没有相应Patch FID收益。把GAN扩大到第3、4步未明显优于只加末步：48.2507对48.1145，没有多种子证据支撑差异显著。</p>
<p>均匀四点联合组目前最好，但它同时改变采样点、PCM边界及fake阶段策略；它不是“仅改变GAN”的消融。训练完整和数值稳定也不等于方法有效。与原SDXL/DMD2仍有明显质量差距。</p>
<p><a href="../coco_uniform_g160_20261025/report.html">完整COCO10K报告</a> · <a href="../coco_uniform_g160_20261025/combined_scores.json">原始分数JSON</a> · <a href="../coco10k_g160_20261020/manifest.json">评测清单及模型配置</a></p>
<h2>2. 后续边界DM及先验实验</h2><table><tr><th>方案</th><th>实现</th><th>核验状态</th><th>质量结论</th></tr>
<tr><td>均匀四点，第3、4步GAN+DMD</td><td>旧clean预测目标，计划与均匀末步联合组对照</td><td>内存/资源问题及队列取消，未完成</td><td>不能作为已完成消融</td></tr>
<tr><td>局部边界DM，无先验</td><td>PCM＋随机中段边界DM＋末步DM，无GAN</td><td>.33.60完成G6，其他任务进入后保护性终止；status残留running不能当作仍运行</td><td>不能充当G160基线</td></tr>
<tr><td>局部边界DM，归一化转向</td><td>段内70%均匀＋30%转向曲线；先验仅用于学生</td><td>.32.81完成G160/fake320，完整续训状态有完成标记</td><td>本次未取得完整COCO10K结果</td></tr>
<tr><td>局部边界DM，原始转角</td><td>与上一组对应，只替换先验曲线</td><td>.32.81最新G105/fake212，正在下一周期中</td><td>训练未完成，尚无完整质量结论</td></tr></table>
<p>旧先验两组合法加噪范围是509–739、259–489、20–239。它们沿用fake阶段轮转，先验只作用于学生。当前A800方案将上界扩到979，fake增加阶段条件，同时对齐student/fake采样分布和阶段配对。因此二者是不同版本的研究方案，不能把跨版本差异单独归因于先验。学生和fake时间权重不同本身不自动证明旧训练数学错误，但不是当前对齐实现。</p>
<p>归一化正式日志只有158条学生更新，是从已验证G2状态续训，合并验证段为160次；原始转角同理，从G2续训。不能把这种日志分段误报为缺少更新。完整存档本次核查到完成标记，未额外重载实验室大体积续训分片。</p>
<p><a href="lab81_snapshot.txt">.32.81实时只读证据</a> · <a href="lab60_history.txt">.33.60历史配置与状态</a></p>
<h2>3. 诊断与工程成果</h2><table><tr><th>工作</th><th>已获得的证据</th><th>使用边界</th></tr>
<tr><td>512条老师轨迹</td><td>128提示词×4种子；归一化转向总体前高后低，未找到可靠独特中段分界</td><td>几何诊断，不能直接当作监督收益证明</td></tr>
<tr><td>逐阶段老师替换</td><td>32提示词×6分支；首步替换最影响全局，末步替换显著改变高频统计</td><td>输出敏感性不等于该阶段增加训练必然有效</td></tr>
<tr><td>早期调参与GAN短训</td><td>学习率/PCM四组各12G、GAN早期四组各24G</td><td>验证流程与短期稳定性，不确定长期最优超参</td></tr>
<tr><td>双40GB卡microbatch16</td><td>16×2卡×累积2=64；fake及学生Adam卸载，完成G160</td><td>不能直接外推为无卸载也可容纳相同批量</td></tr>
<tr><td>四24GB卡无卸载</td><td>冻结教师分片；1×4卡×累积16=64，峰值约19.96GiB/卡，正式训练跑通</td><td>资源受限下的工程可行性，非高速配置</td></tr></table>
<p><a href="../teacher_turning_resample_20261026/results512/report.html">512条老师轨迹</a> · <a href="../stage_intervention_20261026/report.html">阶段替换诊断</a> · <a href="../microbatch_tuning_20261023/report.html">microbatch实验</a></p>
<h2>4. 当前租赁A800与旧结果如何衔接</h2><p>B已完成G160并通过CPU严格权重加载检查；C最近定时快照G83（10:26，非本次实时刷新）。它们采用均匀四点、跨区间边界DM、fake两次且与学生配对、无GAN。B先验900–979；C先验20–100及900–979并集合计30%，补集70%。中段不能使用20–100，B/C的中段采样PDF相同。</p>
<p>B/C是固定区间配额实验：C同时增加低噪声权重并减少高噪声权重，不能称为“只增加低噪声”。这也不是完整DyMD的自适应先验实现。<a href="../a800_BC_paired_quota_20261102/report.html">当前B/C报告</a></p>
<h2>5. 建议的下一步：先回答可归因的问题</h2>
<ol><li><strong>补新版无先验A。</strong>沿用当前B/C全部模型、数据、初始权重、随机种子、步数、阶段配对、CFG、优化器设置及979上界，只把student/fake的再加噪时间改为各自合法范围内均匀采样。预计单A800 G160约7小时（依现有周期速度估算，未执行）。A对B回答高段加强是否有效；A对C回答双端分配是否有效；B对C回答如何分配先验。</li>
<li><strong>先统一评测已有权重，再选GAN实验。</strong>当前B/C及完成的旧归一化先验模型先用同提示词、同种子的固定开发样本检查语义、纹理和伪影；必要时1K仅作筛查，分数只在相同样本量内比较，不能和历史10K FID混比。A/B/C最终仍按历史COCO10K协议评测，保留相同样图。沿用中心Patch FID保证历史连续性，另列固定多位置patch统计或盲看局部裁剪，避免只覆盖图像中心。</li>
<li><strong>做小型2×2，分清先验与GAN。</strong>固定某一种先验P（说明开发集选择依据），比较A、P、A＋末步GAN、P＋末步GAN。若预算只够一个新增GAN组，则先做P＋GAN与P配对，只能说明在P条件下GAN是否有收益，不能声称二者存在协同效应。先采用末步GAN+DMD；历史证据不支持优先投入末步纯GAN或把GAN扩到中段。</li>
<li><strong>正向结果出现后重复种子。</strong>优先重复A与最佳方案，确认收益不是单个训练种子；再讨论增加数据量/更新次数和SD3迁移。若要将贡献表述为教师先验，随后还需与等强度的手工时间分配对照，证明教师统计提供了额外信息。</li></ol>
<p>当前缺口是严格匹配的基线与完成评测，不是继续堆叠更多模块。以上是供讨论的计划，本次不自动启动新训练。正在运行的实验按现有授权继续。</p></html>'''
(out/'report.html').write_text(report,encoding='utf-8')
(out/'scores.json').write_text(json.dumps(scores,ensure_ascii=False,indent=2),encoding='utf-8')
print(out/'report.html')
