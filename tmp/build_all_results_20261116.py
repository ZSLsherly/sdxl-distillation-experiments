"""Consolidate archived experiment evidence; does not connect or start jobs."""
import json, hashlib, html, re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1] / 'output/sdxl-crossdistill'
OUT = ROOT / 'all_experiments_20261116'
OUT.mkdir(exist_ok=True)
sources = set()
def read(p):
    p = ROOT / p
    sources.add(p)
    return json.loads(p.read_text(encoding='utf-8-sig'))

rows = read('fid_pfid_comparison_20261113_side/results_and_recipes.json')
semantic = read('clip_all_20261113/results_all.json')
aliases = {'PCM_DM4':'PCM_DM4_G160', 'A10_GAN':'A_pcm10_patchgan',
           'A_clean':'A_mid_clean_x0', 'H20_GAN':'H20_extra_GAN'}
ids = {r['id']:r['label'] for r in rows}
labkeys = {'dmd_only','gan4','joint4','joint34','joint4_uniform','normalized_u_g160','raw_angle_g160'}
for r in rows:
    m = semantic['metrics'][aliases.get(r['key'],r['key'])]
    r.update(CLIP=m['cosine_x100'], CLIP_n=1024, top1=m['top1_correct'], top5=m['top5_correct'])
    r['training_host'] = '公开权重' if r['family']=='public' else ('实验室' if r['key'] in labkeys else ('H20' if r['key']=='H20_GAN' else 'A800'))
    r['hours'] = r.get('hours')
    r['difference'] = re.sub(r'(?<=基于)\d+|(?<=相比)\d+|(?<=沿用)\d+|(?<=在)\d+(?=基础)', lambda m: '「'+ids[m.group()]+'」',r['difference'])
    r['status'] = '公开参照' if r['family']=='public' else 'G160；FID/Patch 10K；CLIP 1024'
    r['semantic_source'] = 'clip_all_20261113/results_all.json'

new = read('new_models_eval_20261114/summary_four_complete.json')
specs = {
 'A_clean_global':('A · 中段clean＋全局均匀', '以A中段clean为基线，仅把中段加噪支持扩为20–979；student/fake同分布。PCM×1，无先验、无混合、无GAN。'),
 'A_terminal_only':('A · 仅末步学生DM', '以A为基线，仅删除学生中段DM，保留PCM×1及末步DM；fake仍中段一次＋末步一次。无先验、无混合、无GAN。'),
 'A_pcm10_low30':('A×10 · 末步低噪声30%', '以A×10为基线，仅末步student/fake改为20–100占30%、101–979占70%；中段不变，无混合、无GAN。'),
 'A_clean_global_high30_v2':('A · clean全局＋高噪声30%', '以A中段clean＋全局均匀为基线，第2/3/4步student/fake均以900–979占30%、20–899占70%抽样；PCM×1不变，无混合、无GAN。采用修复随机流后的v2。')}
for key, entry in new.items():
    d = entry['details']
    prefix = f"new_models_eval_20261114/snapshots/{entry['host']}/results/{key}"
    complete = read(prefix+'/all_complete.json')
    scores = read(prefix+'/scores.json')[key]
    sem = read(prefix+'/clip_results.json')
    assert sem['revision']==semantic['revision'] and sem['cases']==semantic['cases'], key
    assert sem['n']==1024 and d.get('fid_n',d.get('n'))==10000
    m = sem['metrics'][key]
    for field in ('FID','Patch_FID'):
        assert abs(scores[field]-d[field]) < 1e-6, (key,field)
    rows.append(dict(key=key,label=specs[key][0],difference=specs[key][1],
        FID=d['FID'],Patch_FID=d['Patch_FID'],CLIP=m['cosine_x100'],CLIP_n=1024,
        top1=m['top1_correct'],top5=m['top5_correct'],n=10000,hours=None,
        family='boundary',training_host='A800',evaluation_host=entry['host'],
        status='G160；FID/Patch 10K；CLIP 1024',source=prefix+'/scores.json',semantic_source=prefix+'/clip_results.json'))
assert len(rows)==26 and len({r['key'] for r in rows})==26
lookup = {r['key']:r for r in rows}
for r in rows:
    assert r['n']==10000
    for f in ('FID','Patch_FID','CLIP'):
        assert 0 < r[f] < 100

comparisons = [
 ('A','B','高噪声先验'),('A','C','双端先验'),('C','C_mix1','从G1混入1%'),
 ('C_mix1','DM234_fake3','中段全选，同时增加fake预算'),
 ('C_mix1','PCM_DM4','删除学生中段DM'),('A','A_terminal_only','在无先验A上删除学生中段DM'),
 ('C_mix1','DM1_mid4_fake3','PCM改为首步DM，同时增加fake预算'),
 ('A','A_pcm10','PCM权重1→10'),('A_pcm10','A_pcm10_mix1','PCM×10基础上混入1%'),
 ('A_pcm10','A_pcm10_low30','末步低噪声30%'),('A','A_clean','中段边界改clean目标'),
 ('A_clean','A_clean_global','clean中段改全局加噪'),('A_clean_global','A_clean_global_high30_v2','clean全局增加高噪声30%'),
 ('C_mix1','H20_GAN','额外去噪＋SAM GAN（跨硬件）'),('A_pcm10','A10_GAN','旧DINO/RGB GAN'),
 ('dmd_only','joint4','旧四点末步增加GAN'),('dmd_only','gan4','旧四点末步GAN替代DM'),
 ('normalized_u_g160','raw_angle_g160','局部范围转向先验：归一化→原始')]
deltas = []
for base,exp,note in comparisons:
    a,b=lookup[base],lookup[exp]
    deltas.append(dict(base=base,experiment=exp,change=note,**{f:b[f]-a[f] for f in ('FID','Patch_FID','CLIP')}))

pending = [
 ('H20·第四步直接接SAM GAN','G33中断，已有G32图；无最终学生权重、无完整COCO10K。不能与额外去噪GAN下结论。','实验记录.txt / h20备份记录'),
 ('D·末步高20%＋低10%＋普通70%','G9主动停止；不计入G160表。','实验记录.txt'),
 ('C＋1%·PCM＋末步DM，继续等墙钟','G160结果已入表；后续延长至G171停止，没有独立完整评测。','实验记录.txt'),
 ('C＋1%·仅末步DM＋fake两次均末步＋低噪声30%（11月6日）','有代码/排队记录；11月12日审计指出完成验收与部署状态未闭环。当前未取得可核验G160分数，不冒充完成。与11月13日A×10低30%不同。','project_audit_20261112/findings.json'),
 ('均匀四点·旧版第3/4步GAN','早期尝试未完成；不能拿旧999/919/499/199三四步GAN代替此对照。','历史总表'),
 ('边界无先验早期短训','G6停止；之后A的G160已提供新版匹配无先验基线。','历史总表'),
 ('高30% clean全局v1、早期两次fake全选版','被修正版替代；日志保留，旧结果不纳入排名。','clean_global_high30_20261114/实验记录.txt / 实验记录.txt'),
 ('20K数据训练','数据准备完成记录存在；当前汇总没有已核验的20K正式训练结果。','实验记录.txt'),
 ('老师诊断与性能短测','512条老师轨迹；32提示词×6分支老师替换；microbatch/卸载/SAM精度/VAE批量测试。用于设计与工程，不计入质量benchmark。','teacher_turning_resample_20261026/results512 / stage_intervention_20261026 / h20_mb32_20261103')]
notes = [
 '整理日期2026-11-16。完整评分证据截至11月15日归档，26项：24个自训G160＋2个公开参照。其中A800训练16组、实验室训练7组、H20训练1组；评测服务器与训练服务器分开。',
 '今日A800原SSH入口connect.nma1.seetacloud.com:14892拒绝连接。以下是归档文件已核实结果，不声称是今日实时队列状态。本次没有启动训练、评测、释放实例或修改训练代码。',
 'FID及Patch FID均为项目固定COCO2014 10,000项；Patch FID取中心299×299区域。它是特征分布指标，不是像素MSE，也不能代表全图每个位置的纹理。',
 'CLIP为同1024提示词/种子、OpenAI ViT-B/32余弦×100。新四组已逐项核对cases与revision一致。Top1为当前候选caption池内的检索计数，不能当通用语义正确率。',
 '训练为原LAION衍生10K，COCO是评测集。G160指160次学生更新，batch64对应10240次学生样本使用，不是160个epoch。fake的样本使用与更新另计。',
 '时长仅沿用已有日志汇总的训练小时，不含完整COCO评测；各组硬件、诊断开销不同，不能直接当训练提速比。未取得完整起止时间的组留空，不由完成时间或短测外推。',
 '大多数近期实验只保留学生推理权重，没有优化器/fake状态；可以评测不等于可原样续训。原C缺历史fake留出loss，不能用训练loss或新组混合前探针补造该基线。',
 '所有指标为单次训练结果；没有多随机种子置信区间。报告中改善指本次测值变化。官方DMD2训练预算不同，仅作质量参照。',
 '局部范围转向实验与A/B/C跨范围实验同时存在加噪范围、fake阶段轮转/配对等差异；跨系列排名不能归因于先验单个因素。',
 '低噪声fake加权仍沿用min(1,10σ²)。抽样概率30%不等于最终梯度权重30%；它是后续解释低噪声消融时需要保留的因素。']
conclusions = [
 '最稳的下一步方向：保留PCM，优先研究末步DM。C＋1%去中段后FID25.656→22.129、Patch32.967→28.024；A去中段后FID25.617→22.437、Patch33.377→30.630。两套背景方向一致，仍只到G160。',
 '教师先验尚无全面优势：B相对A的FID和Patch都变差；C只略改善Patch（−0.229），FID变差（＋0.426）。1%混合在C上变化小，在PCM×10上基本持平。',
 'GAN的局部纹理分布收益值得保留：H20额外去噪＋SAM GAN相对C＋1%的Patch下降7.298，FID近乎不变，CLIP下降0.158。尚缺完成的直接终点GAN，以及“额外去噪但无GAN”匹配消融，不能把全部收益唯一归于GAN。',
 '当前自训最低FID为C＋1%仅末步学生DM（22.129）；最低Patch为H20 GAN（25.669），局部原始转角无GAN也达到25.899。两者训练方案不同，差0.230不足以宣称GAN全面胜出。',
 '中段clean目标提高CLIP但伤害Patch：A→clean，CLIP＋0.389、Patch＋13.139；扩大clean加噪范围能部分修复，却仍比A的Patch差。不能单凭CLIP选择clean方案。',
 'PCM×10末步低噪声30%：相对A×10，Patch35.230→31.538，但FID24.641→24.988，CLIP31.453→31.350，属于明确取舍。clean全局高噪声30%这次没有带来FID/Patch收益。',
 '当前自训组整体质量仍未达到公开SDXL/DMD2：最佳自训FID22.129对18.454；最佳自训Patch25.669对20.286。CLIP均值接近并不等于综合语义/画质达到公开模型；同一1024候选池检索Top1，SDXL/DMD2为804/808，A/B/C为751/764/765。']

order = ['sdxl','dmd2']+[r['key'] for r in rows if r['key'] in labkeys]+[r['key'] for r in rows if r['training_host']=='A800']+['H20_GAN']
ordered = [lookup[k] for k in order]
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
plt.rcParams['axes.unicode_minus']=False
fig,axs=plt.subplots(1,3,figsize=(16,13),sharey=True,gridspec_kw={'width_ratios':[1,1,1]})
colors={'公开权重':'#64748b','实验室':'#228b79','A800':'#3478bd','H20':'#d48528'}
for ax,field,title in zip(axs,['FID','Patch_FID','CLIP'],['FID ↓ · COCO10K','Patch FID ↓ · 中心299','CLIP ↑ · 固定1024']):
    ax.barh(range(26),[r[field] for r in ordered],color=[colors[r['training_host']] for r in ordered],height=.64)
    ax.set_title(title); ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    ax.set_xlim((0, max(r[field] for r in ordered)*1.17) if field!='CLIP' else (30.8,32.32))
    for i,r in enumerate(ordered): ax.text(r[field]+(.06 if field!='CLIP' else .006),i,f'{r[field]:.3f}',va='center',fontsize=8)
    for s in ['right','top']:ax.spines[s].set_visible(False)
axs[0].set_yticks(range(26),[r['label'] for r in ordered],fontsize=9)
axs[0].invert_yaxis()
fig.suptitle('全部已完成评测：24个自训G160＋2个公开参照',fontsize=16)
fig.text(.5,.012,'灰：公开  绿：实验室  蓝：A800  橙：H20 ｜ CLIP横轴截断；不比较条形长度 ｜ 单次训练结果',ha='center',fontsize=10)
fig.tight_layout(rect=[0,.035,1,.97]);fig.savefig(OUT/'all_scores.png',dpi=155);plt.close(fig)

def fmt(v): return '—' if v is None else f'{v:.3f}'
def esc(x):return html.escape(str(x))
def table(rs,detail=False):
    header=['实验','训练来源','FID ↓','Patch FID ↓','CLIP ↑','检索Top1/1024','训练小时']+(['配置差异'] if detail else [])
    h='<table><thead><tr>'+''.join('<th>'+s+'</th>' for s in header)+'</tr></thead><tbody>'
    for r in rs:
        vals=[r['label'],r['training_host'],fmt(r['FID']),fmt(r['Patch_FID']),fmt(r['CLIP']),str(r['top1']),fmt(r['hours'])]+([r['difference']] if detail else [])
        h+='<tr>'+''.join('<td>'+esc(s)+'</td>' for s in vals)+'</tr>'
    return h+'</tbody></table>'
parts=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>全部实验汇总 · 2026-11-16</title>',
 '<style>body{font:16px/1.65 system-ui,"Microsoft YaHei";max-width:1540px;margin:32px auto;padding:0 24px;background:#f6f8fb;color:#172333}h1{font-size:30px}h2{margin-top:36px}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{border:1px solid #dbe2eb;padding:9px;text-align:left}th{background:#eaf0f8;position:sticky;top:0}td:nth-child(1){min-width:160px}td:last-child{max-width:520px}li{margin:9px 0}.box{background:white;border-left:5px solid #3478bd;padding:12px 20px;margin:18px 0}img{width:100%}code{overflow-wrap:anywhere}.scroll{overflow:auto}a{color:#2466a6}</style>',
 '<h1>全部实验结果汇总 · 2026-11-16</h1><div class="box"><b>26项完整评测：24组自训＋2组公开参照。</b><br>包含A800的16组；新增四组均已补齐COCO10K和CLIP1024。当前A800连接失败，运行状态不以历史记录代替。</div>',
 '<h2>结果判断</h2><ol>'+''.join('<li>'+esc(x)+'</li>' for x in conclusions)+'</ol>',
 '<h2>全量图</h2><p>FID/Patch越低越好，CLIP越高越好；三种指标不合成单一总分。</p><img src="all_scores.png">',
 '<h2>A800：全部16组</h2><div class="scroll">'+table([r for r in rows if r['training_host']=='A800'],True)+'</div>',
 '<h2>实验室、H20与公开参照</h2><div class="scroll">'+table([r for r in ordered if r['training_host']!='A800'],True)+'</div>',
 '<h2>配对变化量（实验−基线）</h2><p>负的FID/Patch表示改善；CLIP正值表示提高。多变量实验已标注。</p><table><tr><th>基线 → 实验</th><th>改变</th><th>ΔFID</th><th>ΔPatch</th><th>ΔCLIP</th></tr>']
for d in deltas:
    vals=[lookup[d['base']]['label']+' → '+lookup[d['experiment']]['label'],d['change']]+[f'{d[f]:+.3f}' for f in ['FID','Patch_FID','CLIP']]
    parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in vals)+'</tr>')
parts+=['</table><h2>未完成、替代与诊断实验</h2><table><tr><th>实验</th><th>最后可核验状态</th><th>来源</th></tr>']
for p in pending:parts.append('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in p)+'</tr>')
parts+=['</table><h2>口径与证据限制</h2><ul>'+''.join('<li>'+esc(n)+'</li>' for n in notes)+'</ul>',
 '<h2>原始证据与可复查文件</h2><p><a href="results.json">合并结果JSON</a> · <a href="实验结果汇总.txt">TXT记录</a> · <a href="source_sha256.json">本次引用源文件SHA-256</a></p><ul>']
for p in sorted(sources):parts.append(f'<li><a href="../{esc(p.relative_to(ROOT).as_posix())}">{esc(p.relative_to(ROOT).as_posix())}</a></li>')
parts+=['<li><a href="../new_models_eval_20261114/四组完成与校验记录.txt">四组rclone传输及SHA-256记录</a></li></ul></html>']
(OUT/'report.html').write_text('\n'.join(parts),encoding='utf-8')
(OUT/'results.json').write_text(json.dumps(dict(date='2026-11-16',rows=ordered,deltas=deltas,notes=notes,pending=pending),ensure_ascii=False,indent=2),encoding='utf-8')
txt='全部实验汇总 2026-11-16\n\n'+'\n'.join(notes)+'\n\n结论\n'+'\n'.join(conclusions)+'\n\n完整结果：名称 | 训练来源 | FID | Patch FID | CLIP1024 | 训练小时\n'
for r in ordered:txt+=' | '.join([r['label'],r['training_host']]+[fmt(r[f]) for f in ['FID','Patch_FID','CLIP','hours']])+'\n配置：'+r['difference']+'\n来源：'+r['source']+'；语义：'+r['semantic_source']+'\n'
txt+='\n未完成与诊断\n'+'\n'.join(' | '.join(p) for p in pending)
(OUT/'实验结果汇总.txt').write_text(txt,encoding='utf-8')
(OUT/'source_sha256.json').write_text(json.dumps([{ 'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(sources)],ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'rows':len(rows),'a800':sum(r['training_host']=='A800' for r in rows),'sources':len(sources),'report':str(OUT/'report.html')},ensure_ascii=False))
