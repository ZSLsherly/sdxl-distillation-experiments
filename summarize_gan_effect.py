import json,csv,statistics
from pathlib import Path
from html import escape
p=Path(__file__).resolve().parent/'output/sdxl-crossdistill/benchmark_curves_four_20261018'
rows=json.loads((p/'summary.json').read_text())
with (p/'per_image.csv').open(encoding='utf-8') as f:images=list(csv.DictReader(f))
variants=['dmd_only','joint4','gan4','joint34']
names={'dmd_only':'纯DMD（首步TM）','joint4':'第4步GAN+DMD','gan4':'第4步纯GAN','joint34':'第3、4步GAN+DMD'}
g118={v:next(r for r in rows if r['variant']==v and r['student_updates']==118) for v in variants}
details={v:{int(r['original_index']):r for r in images if r['variant']==v and int(r['student_updates'])==118} for v in variants}
effect=[]
for v in variants[1:]:
    base=g118['dmd_only'];gan=g118[v]
    win_niqe=sum(float(details[v][i]['niqe'])<float(details['dmd_only'][i]['niqe']) for i in details[v])
    win_clip=sum(float(details[v][i]['clip_cosine'])>float(details['dmd_only'][i]['clip_cosine']) for i in details[v])
    effect.append({'variant':v,'NIQE_change_GAN_minus_DMD':gan['niqe']-base['niqe'],'NIQE_reduction_percent':(base['niqe']-gan['niqe'])/base['niqe']*100,'CLIP_change_GAN_minus_DMD':gan['clip_cosine']-base['clip_cosine'],'NIQE_better_prompts':win_niqe,'CLIP_better_prompts':win_clip})
window={v:{metric:statistics.mean(r[metric] for r in rows if r['variant']==v and 96<=r['student_updates']<=118) for metric in ['niqe','clip_cosine']} for v in variants}
out={'G118':g118,'effect_vs_DMD':effect,'G96_118_checkpoint_means':window,'note':'12 checkpoints reuse the same 4 prompts; window means are descriptive, not 48 independent samples or a significance test.'}
(p/'gan_effect_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
table='<h2>G118：相对纯DMD的变化</h2><table><tr><th>方案</th><th>NIQE变化（负数更好）</th><th>NIQE改善比例</th><th>CLIP变化（正数更好）</th><th>NIQE更好的样本</th><th>CLIP更好的样本</th></tr>'
for r in effect:table+=f'<tr><td>{names[r["variant"]]}</td><td>{r["NIQE_change_GAN_minus_DMD"]:+.4f}</td><td>{r["NIQE_reduction_percent"]:+.2f}%</td><td>{r["CLIP_change_GAN_minus_DMD"]:+.4f}</td><td>{r["NIQE_better_prompts"]}/4</td><td>{r["CLIP_better_prompts"]}/4</td></tr>'
table+='</table><h2>G96–118曲线区间均值</h2><p>用于查看后期趋势，12个检查点沿用相同4个样本，不是48个独立样本。</p><table><tr><th>方案</th><th>NIQE ↓</th><th>CLIP ↑</th></tr>'
for v in variants:table+=f'<tr><td>{names[v]}</td><td>{window[v]["niqe"]:.4f}</td><td>{window[v]["clip_cosine"]:.4f}</td></tr>'
table+='</table><h2>G118逐图分数</h2><p>每格为 NIQE ↓ / CLIP cosine ↑。</p><table><tr><th>样本</th>'+''.join(f'<th>{names[v]}</th>' for v in variants)+'</tr>'
for i in sorted(details['dmd_only']):
    table+=f'<tr><td>{i}</td>'+''.join(f'<td>{float(details[v][i]["niqe"]):.3f} / {float(details[v][i]["clip_cosine"]):.4f}</td>' for v in variants)+'</tr>'
table+='</table>'
html=(p/'report.html').read_text(encoding='utf-8').replace('<h2>同G118图片对照</h2>',table+'<h2>同G118图片对照</h2>')
(p/'report.html').write_text(html,encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
