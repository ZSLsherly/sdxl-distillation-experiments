"""Create a presentation archive from measured results without changing scores."""
from pathlib import Path
import json
import hashlib
import html

root = Path(__file__).resolve().parents[1] / 'output/sdxl-crossdistill'
src = root / 'all_experiments_20261116/results.json'
out = root / 'interview_archive_20261116'
out.mkdir(exist_ok=True)
data = json.loads(src.read_text(encoding='utf-8'))
r = {x['key']:x for x in data['rows']}
pairs = [
    ('A','A_terminal_only','A：去掉学生中段DM'),
    ('C_mix1','PCM_DM4','C＋1%：去掉学生中段DM'),
    ('C_mix1','H20_GAN','增加额外去噪＋SAM GAN'),
    ('A_pcm10','A10_GAN','增加末步DINO/RGB GAN'),
    ('A','A_pcm10','PCM权重1→10'),
    ('C_mix1','DM1_mid4_fake3','首步PCM改DM，同时fake两次→三次'),
    ('A','B','边界DM：高噪声30%先验'),
    ('A_clean_global','A_clean_global_high30_v2','clean全局DM：高噪声30%先验'),
]
lines = [
    'SDXL分阶段蒸馏：面试展示存档（含待核对补充）',
    '整理日期：2026-11-16；证据截至2026-11-15归档。',
    '',
    '公司端补充实验结论（用户提供，待核对）',
    '用户说明：另有使用Claude开展的实验，材料保存在公司服务器，目前无法提供，之后重新审理。下列内容记录用户补充的结论；本地尚无对应新配置、数值或日志，未纳入已有26项分数表。',
    '1. 用户报告：删除中段DM后的改善与PCM loss比例降低有关，新增对照支持这一解释。待核对公司端的实际损失组合、归一化方式及PCM权重；本地原对照本身未改变PCM系数。',
    '2. 用户报告：补充实验显示PCM与GAN均有明显效果。待补充各自的匹配基线及效果指标；本地已确认的GAN Patch FID变化见下文。',
    '3. 用户报告：纯高噪声先验在训练前期效果不明显。待核对“前期”的更新区间、采样配额及分阶段评测；本地现有总表是G160终点评测。',
    '补充结论来源为用户陈述，尚未独立核验。后续取得公司端材料后再更新相关数值和结论；本存档不包含推测或代填的新实验分数。',
    '',
    '项目目标',
    '研究四步SDXL蒸馏中，不同阶段监督、加噪先验和GAN纹理监督的作用；用匹配对照判断哪些设计值得保留。',
    '',
    '实验规模与口径',
    '已汇总24个自训G160模型及2个公开参照，包含16组A800实验。训练为LAION衍生10K；G160是160次学生更新，不是160个epoch。近期有效batch64。',
    'FID/Patch FID：固定COCO2014 10K；Patch取中心299×299。CLIP：固定1024条提示词及种子，ViT-B/32余弦×100。以下均为原始实测值，单次训练。',
    '',
    '发现一：精简中段监督，在两种配置下都改善了G160结果',
    'A去掉学生中段DM后，FID25.617→22.437，Patch33.377→30.630；C＋1%同样去掉中段DM后，FID25.656→22.129，Patch32.967→28.024。两组保留首步PCM、末步DM及fake的中段更新。',
    '可以展示的结论：在当前训练预算下，中段边界DM并未带来额外收益；精简学生监督是有效方向。',
    '机制假设：可能涉及不同目标的梯度竞争及相对贡献变化。尚不能归因于PCM loss比例降低。PCM系数未减小；删除一个DM项也不能直接推出PCM实际梯度占比降低，需要匹配的权重/梯度对照。',
    '',
    '发现二：GAN带来明确的Patch FID改善；PCM的作用存在指标取舍',
    'C＋1%加入额外去噪＋SAM GAN后，Patch32.967→25.669，降低约22.1%；FID25.656→25.667，基本持平；CLIP31.636→31.477。',
    '另一套A×10配置增加DINO/RGB GAN后，Patch35.230→33.919，FID24.641→24.821。两个背景都观察到局部分布指标改善，但没有全面改善所有指标。',
    'PCM权重1→10时，FID25.617→24.641，Patch33.377→35.230，CLIP31.675→31.453。首步PCM替换为DM的现有实验更差，但同时增加了fake更新次数。',
    '可以展示的结论：PCM是值得保留的首步监督；GAN模块在现有实验中改善局部纹理分布。PCM独立因果作用、GAN与额外去噪各自的贡献仍需进一步拆分。',
    '',
    '发现三：只提高高噪声采样概率，没有在当前预算表现出收益',
    'A→B：FID25.617→26.274，Patch33.377→36.195，CLIP基本相同。clean全局配置增加高噪声30%：FID23.737→24.828，Patch42.240→42.824。',
    '可以展示的结论：两种配置的G160终点评测都没有支持单独强化高噪声。这里“纯高噪声先验”仍是高区30%、普通区70%，并非全部样本只取高噪声。',
    '目前这些终点评测不能定位收益出现在哪个训练时期，因此不写成“只在前期无效”或“后期一定有效”。',
    '',
    '面试口述版本',
    '我围绕四步SDXL做了阶段监督、教师先验和GAN的系统对照。最一致的发现是：在两套背景下，保留首步PCM和末步DM、移除学生中段DM，都改善了FID与Patch FID。随后引入GAN，局部Patch FID在两套实现里都有下降，其中SAM方案下降约22%，但全局FID基本持平。这说明阶段分工和纹理监督值得继续，而单独加大高噪声采样没有显示收益。我把指标改善和机制解释分开：目前有结果支持精简监督，梯度竞争的解释还需要进一步验证。',
    '',
    '对照明细（原始分数）',
]
tr = []
for a,b,label in pairs:
    x,y = r[a],r[b]
    vals = [label] + [f"{x[k]:.3f} → {y[k]:.3f}" for k in ('FID','Patch_FID','CLIP')]
    lines.append(' | '.join(vals))
    tr.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in vals)+'</tr>')
lines += [
    '',
    '公开参照：SDXL50步 FID18.533/Patch20.286；DMD2四步 FID18.454/Patch20.437。自训最佳FID22.129，最佳Patch25.669；尚未达到公开参照的综合质量。',
    '所有26项原始分数保存在本目录results.json，与源文件逐字节相同。完整源报告：'+str(src.parent/'report.html'),
    '本存档只更新表达和组织，不修改分数、不补造未完成实验或未测机制。',
]
(out/'展示说明.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(out/'results.json').write_bytes(src.read_bytes())
sha = hashlib.sha256(src.read_bytes()).hexdigest()
assert hashlib.sha256((out/'results.json').read_bytes()).hexdigest()==sha
manifest = dict(source=str(src),destination=str(out/'results.json'),sha256=sha,byte_identical=True,n_results=len(data['rows']))
(out/'存档核验.txt').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>SDXL阶段蒸馏·面试展示</title>',
 '<style>body{max-width:1080px;margin:38px auto;padding:0 24px;font:16px/1.75 system-ui,"Microsoft YaHei";color:#172333}h1{font-size:29px}h2{font-size:21px;color:#245b8f;margin-top:32px}p{margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #d4dce7;padding:10px}th{background:#edf3f9}a{color:#245b8f}</style>',
 '<h1>'+lines[0]+'</h1>']
heads={'公司端补充实验结论（用户提供，待核对）','项目目标','实验规模与口径','发现一：精简中段监督，在两种配置下都改善了G160结果','发现二：GAN带来明确的Patch FID改善；PCM的作用存在指标取舍','发现三：只提高高噪声采样概率，没有在当前预算表现出收益','面试口述版本'}
for line in lines[1:lines.index('对照明细（原始分数）')]:
    if line:parts.append(('<h2>'+html.escape(line)+'</h2>') if line in heads else '<p>'+html.escape(line)+'</p>')
parts+=['<h2>对照明细</h2><table><tr><th>改变</th><th>FID ↓</th><th>Patch FID ↓</th><th>CLIP ↑</th></tr>']+tr+['</table>']
for line in lines[-4:]:parts.append('<p>'+html.escape(line)+'</p>')
parts+=['<p><a href="展示说明.txt">TXT展示说明</a> · <a href="results.json">全部原始结果</a> · <a href="存档核验.txt">SHA-256核验</a></p></html>']
(out/'report.html').write_text('\n'.join(parts),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
