from pathlib import Path
import json,html,re
from datetime import datetime
p=Path('output/sdxl-crossdistill');o=p/'all_experiments_20261104'
def J(f):return json.loads((p/f).read_text(encoding='utf-8-sig'))
def table(headers,rows):
 return '<table><thead><tr>'+''.join('<th>'+html.escape(str(s))+'</th>' for s in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(s))+'</td>' for s in row)+'</tr>' for row in rows)+'</tbody></table>'
def link(d,name):return f'<a href="../{d}">{name}</a>'
a=J(o.name+'/a800.json');h=J(o.name+'/h20.json');old=J('lab_results_review_20261102/scores.json')
sem=J('semantic_four_1024_20261103/results.json')['metrics']; sm128=J('semantic_abc_20261103/results.json')['metrics']
srows=J('benchmark_final_20261020/summary.json'); final=J('benchmark_final_20261020/analysis.json')['G160']
rows=[]; dash='—'
def row(name,hw,g,score=None,clip=dash,t=dash):
 rows.append([name,hw,g,clip,*([f"{score[k]:.4f}" for k in ['FID','Patch_FID']] if score else [dash,dash]),t])
def elapsed(start,end):
 def d(s):return datetime.strptime(s,'%Y%m%dT%H%M%S%z')
 return (d(end)-d(start)).total_seconds()/3600
ref_sem={}
ref_names={'sdxl':'原始 SDXL，50步','dmd2':'官方 DMD2，4步'}
cohort=J('semantic_four_1024_20261103/results.json')
for k,n in ref_names.items():
 j=J(o.name+'/'+k+'_results.json')
 assert j['revision']==cohort['revision'] and j['n']==len(j['cases'])==1024 and j['cases']==cohort['cases']
 v=j['metrics'][k]; pc=j['per_case'][k]
 assert len(pc['alignment'])==len(pc['rank'])==1024 and abs(sum(pc['alignment'])/1024-v['cosine_x100'])<1e-5
 assert sum(r==1 for r in pc['rank'])==v['top1_correct'] and sum(r<=5 for r in pc['rank'])==v['top5_correct']
 ref_sem[n]=v
 row(n,'公开权重','参照',old[k],clip=f"{v['cosine_x100']:.3f} 〔COCO1024〕",t='不适用')
for k,n in [('dmd_only','旧四点 · 纯 DMD'),('gan4','旧四点 · 末步纯 GAN'),('joint4','旧四点 · 末步 GAN+DMD'),('joint34','旧四点 · 第3、4步 GAN+DMD')]:
 z=[r for r in srows if r['variant']==k];hh=elapsed(z[0]['timestamp'],z[-1]['timestamp'])
 row(n,('2' if k=='gan4' else '4')+'×40G 实验室卡',160,old[k],f"{final[k]['clip_cosine']*100:.3f} 〔留出4〕",f'{hh:.2f} h†')
u=J('uniform10k_results_20261025/results.json')
stamp=lambda i:Path(u['evals'][str(i)]['path']).name.split('_')[0]
new_sem=p/'semantic_old_uniform_20261104/results.json'
new_metrics=json.loads(new_sem.read_text(encoding='utf-8'))['metrics']['joint4_uniform'] if new_sem.exists() else None
row('均匀四点 · 末步 GAN+DMD（旧 clean 目标）','2×40G 实验室卡',160,old['joint4_uniform'],clip=(f"{new_metrics['cosine_x100']:.3f} 〔COCO1024〕" if new_metrics else dash),t=f'{elapsed(stamp(0),stamp(160)):.2f} h†')
modern=[]
def add(k,n,device,db,run,score):
 z=db[run];ss=db[score+' scores'][k] if score else None
 v=f"{sem[k]['cosine_x100']:.3f} 〔COCO1024〕" if k in sem else dash
 row(n,device,z['g'],ss,v,f"{z['hours']:.2f} h"+(' 已用' if z['g']<160 else ''));modern.append(rows[-1])
base='turning_BC_paired_quota_20261102/'; win='bc_winner_late_mix_20261102/'
for k,n,r,s in [('A','A · 均匀无先验','uniform_control','coco10k_A'),('B','B · 高噪声先验','B','coco10k'),('C','C · 双端先验','C','coco10k')]:add(k,n,'1×A800',a,base+r,win+s)
add('C_mix1','C＋1% · 无 GAN 基线','1×A800',a,win+'late_mix',win+'coco10k_C_mix1')
for k,n,r in [('DM234_fake3','PCM＋DM2/3/4 全选，fake×3','c_mix1_dm234_fake3_20261103'),('DM1_mid4_fake3','DM1＋随机中段＋DM4，fake×3','c_mix1_first_dm_fake3_g160')]:
 add(k,n,'1×A800',a,r+'/model','c_mix1_dm_pair_coco10k/'+k)
r='h20_C_mix1_terminal_GAN_G160'
add('C_mix1_terminal_GAN','C＋1% · 额外去噪末步 GAN','1×H20',h,r,r+'_coco10k')
add('', 'C＋1% · 第四步直接 GAN','1×H20',h,'h20_C_mix1_direct_terminal_GAN_G160',None)
add('', 'PCM＋仅末步学生 DM，中段 fake 保留','1×A800',a,'c_mix1_pcm_dm4_matched_time/model',None)
headers=['实验','训练硬件','学生更新 G','语义 CLIP×100 ↑','FID ↓','Patch FID ↓','实测训练时间']
assert sum(r[4]!=dash for r in rows)==14
body='<h1>SDXL 四步蒸馏实验总表</h1><p>2026-11-04 15:29 训练进度快照，18:38 补录两组公开模型语义分。14 组完成固定 COCO2014 10K 评测，其中 12 组自训、2 组公开模型。另有 2 组正式训练进行中。</p>'
body+='<h2>正式实验与公开参照</h2>'+table(headers,rows)
body+='<p>— 表示未取得对应指标。FID 和 Patch FID 均为完整 10,000 张；后者使用原图中心 299×299。表中的 batch FID 按项目已有的 Patch FID 理解。训练用 LAION 衍生 10K，COCO 为独立评测。旧四点为 999/919/499/199，均匀四点为 999/749/499/249。旧组和新组的 DM 目标、fake 更新策略不同。</p><p>时间单位是实际墙钟小时，不含事后 COCO10K。† 由固定图 G0 到 G160 时间戳求差，含评估等间隔；其余完成组由 G1 周期开始至 G160 更新结束，进行中组截至最近完整更新。不同卡数的墙钟时间不等于 GPU 小时。无中段组还会按约 6.90 h 取等时权重。</p>'
body+='<h2>语义评测</h2><p>CLIP ViT-B/32 图文余弦×100。下表使用相同 COCO 提示词与种子；1,024 条包含原 128 条。旧组的 4 张和短训的 16 张为其他留出样本，分数不能横向排名。</p>'+table(['组','COCO1024 CLIP','Top‑1 /1024','Top‑5 /1024','COCO128 CLIP','Top‑1 /128'],[[k,f"{v['cosine_x100']:.3f}",v['top1_correct'],v['top5_correct'],f"{sm128[k]['cosine_x100']:.3f}" if k in sm128 else dash,sm128[k]['top1_correct'] if k in sm128 else dash] for k,v in {**ref_sem,**sem,**({'旧均匀四点末步 GAN+DMD':new_metrics} if new_metrics else {})}.items()])
body+='<h2>未取得完整跑分的训练及早期试验</h2>'
early=[]; met=J('hparam_pilot/assessment/result.json')['metrics']
for k,v in met.items():
 if 'training' in v:
  e=v['training'];early.append([f"早期学习率 {e['config']['lr_student']:.1e}，PCM 权重 {e['config']['pcm_weight']:g}",'G12',f"{100*v['clip_cosine_mean']:.3f} 〔16〕",dash,dash,f"{e['train_seconds']/60:.1f} 分"])
for k,n,cl in [('dmd','DMD 对照',24.92),('gan4','末步仅 GAN',25.00),('joint4','末步联合',25.01),('joint34_matched','后两步 GAN、匹配次数',25.02)]:
 e=J(f'gan_stage_pilot_v2/{k}/summary.json')
 early.append(['早期 GAN · '+n,'追加 G24',f'{cl:.2f} 〔16〕',dash,dash,f"{e['train_seconds']/60:.1f} 分"])
for k,v in met.items():
 if 'training' not in v and 'clip_cosine_mean' in v:early.append(['早期推理参照 · '+k,'仅推理',f"{v['clip_cosine_mean']*100:.3f} 〔16〕",dash,dash,'不适用'])
early+= [[n,g,dash,dash,dash,t] for n,g,t in [
 ('D · 末步高20%/低10%/普通70%','G9 停止','0.39 h'),
 ('局部边界 DM · 无先验','G6 后停止',dash),('局部边界 DM · 归一化转向','G160','末轮 1,201 秒'),('局部边界 DM · 原始转角','至少 G105¹','当时末轮 1,186 秒'),
 ('均匀四点 · 旧版第3、4步联合 GAN','4090 尝试／A100 队列取消',dash),
 ('早期 B/C · 掩膜先验／固定配额／fake 与学生同分布但未配对','迭代版，已替换',dash),('原拟两组 24G 权重混合短测','撤下',dash)]]
body+=table(['实验','进度','小样本 CLIP×100','FID','Patch FID','训练时间或末轮速度'],early)
body+='<p>¹ 实验室 .32.81 本次 SSH 超时。局部先验两组沿用 11 月 2 日存档；G105 是原始转角组已核验的下限。末轮速度未外推为实测总时长。顺序采样的 B/C 迭代版不计入正式配对 B/C。</p>'
body+='<h2>性能、机制和数据实验</h2>'
short=[]
for v in J('a800_B_batch_probe_20261101/compare/summary.json')['results']:
 b=v.get('result') or {};cy=b.get('cycle_seconds',[])[b.get('warmup_cycles',0):]
 short.append(['A800 · '+v['label'],v['status'],f"{sum(cy)/len(cy):.2f} 秒/G" if cy else dash])
for k,v in J('h20_mb32_20261103/comparison.json').items():
 if isinstance(v,dict) and 'steady_mean_seconds' in v:short.append(['H20 · '+k,'3G、6fake、3D',f"{v['steady_mean_seconds']:.2f} 秒/G，整卡 {v['device_sampled_peak_GiB']:.2f} GiB"])
short.extend([['H20 · UNet32，FP32 SAM 整批／空间分块','两种均 G0 OOM','分块曾完成 2fake＋1D'],['H20 · UNet16，FP32 SAM 整批16','G0 OOM','2fake＋1D'],['H20 · FP32 SAM4，VAE2','G3；正式选 VAE1','365.56 秒/G，reserved 93.26 GiB'],['H20 · 第四步直接 GAN','G3、6fake、3D','348.24 秒/G，整卡 84.20 GiB'],['双 40G 实验室卡 · microbatch 调优','每卡16通过，32 OOM','每卡16稳定 208.00 秒/G'],['两张 48G 4090 · 无卸载','两个先验方案各 G3，有效 batch 8','约 110 秒/G'],['四张 24G 4090 · 分片教师','有效 batch64，完整更新、评估、存档通过','约 20 GiB/卡'],['SDXL 教师轨迹','128 提示词×4 种子＝512 条','按单位距离归一化后总体前高后低'],['教师分阶段替换','32 提示词×6 分支＝192 张','首步影响全局，末步影响高频与纹理'],['20K 数据准备','20,000 训练＋32 留出','本表的正式分数均来自 10K 训练方案']])
body+=table(['实验','范围／状态','实测数据或观察'],short)
body+='<h2>原始记录与分项报告</h2><ul>'
for path,title in [('lab_results_review_20261102/report.html','实验室训练、局部先验及中止实验'),('benchmark_final_20261020/report.html','旧四点 G160 留出图与语义曲线'),('uniform10k_results_20261025/report.html','均匀四点末步 GAN'),('coco_uniform_g160_20261025/combined_scores.json','早期完整 COCO 分数'),('semantic_four_1024_20261103/report.html','COCO1024 语义与配对区间'),('semantic_abc_20261103/report.html','COCO128 语义'),('semantic_old_uniform_20261104/report.html','旧均匀四点 COCO1024 语义补测'),('hparam_pilot/report.html','四组超参短训'),('gan_stage_pilot_v2/report.html','四组 GAN 短训'),('side_teacher_perf_20261029/report.html','3090 教师复制／分片／通信计时'),('stage_intervention_20261026/report.html','阶段替换'),('teacher_turning_resample_20261026/results512/report.html','512 条轨迹'),('turning_ablation_20261027/report.html','转向实验与短测'),('h20_mb32_20261103/comparison.json','H20 短测与精度'),('microbatch_tuning_20261023/report.html','A100 双卡批量调优')]:
 if (p/path).exists():body+='<li>'+link(path,title)+'</li>'
body+='</ul><p>本次远程核查摘录：<a href="a800.json">A800</a> · <a href="h20.json">H20</a></p>'
style='body{font:15px/1.55 system-ui,"Microsoft YaHei";color:#203345;max-width:1280px;margin:30px auto;padding:0 18px}h1,h2{color:#164e57}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{text-align:left;padding:8px;border-bottom:1px solid #d7e1e5}th{background:#e7f1f2}tr:nth-child(even){background:#f7f9fa}a{color:#126677}'
(o/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SDXL 实验总表</title><style>'+style+'</style>'+body+'</html>',encoding='utf8')
(o/'实验总表.txt').write_text('\n'.join(['2026-11-04 15:29 训练进度快照，18:38 补录公开模型语义分。语义列为 CLIP 余弦×100，括号为样本协议；FID 和中心299 Patch FID 均是固定 COCO10K。',*['\t'.join(map(str,r)) for r in [headers,*rows]],'† G0 到 G160 固定图时间戳，其余为完整更新跨度，均不含事后评测。',*[f"{n} COCO1024 检索 Top-1 {v['top1_correct']}/1024，Top-5 {v['top5_correct']}/1024" for n,v in ref_sem.items()],'早期训练：实验 / 进度 / 小样本语义 / FID / Patch FID / 用时',*['\t'.join(map(str,r)) for r in early],'诊断与性能',*['\t'.join(map(str,r)) for r in short]]),encoding='utf-8-sig')
print(len(rows),len(early),len(short));print('\n'.join(' | '.join(map(str,r)) for r in rows))
