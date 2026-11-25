import hashlib,html,json,statistics,zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[1];src=root/'sdxl_research'
out=root/'output/sdxl-crossdistill/microbatch_tuning_20261023'
r=json.loads((out/'evidence/results.json').read_text())
resume=json.loads((out/'evidence/resume_record.json').read_text())
good=r['microbatch16_20261023'];s=good['status']
assert s['status']=='complete' and resume['restored_progress']=={'gcount':2,'fcount':10,'dcount':10}
for k,v in resume['source_sha256'].items():assert hashlib.sha256((src/k).read_bytes()).hexdigest()==v,k
old=[x for x in r['original_mb4']['rows'] if x['phase']=='student_done']
new=[x for x in good['rows'] if x['phase']=='student_done']
ratio=old[1]['cycle_seconds']/new[1]['cycle_seconds']
manifest=json.loads((src/'source_manifest.json').read_text())
with zipfile.ZipFile(out/'sdxl_research_repack.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in list(manifest)+['source_manifest.json']:z.write(src/name,'sdxl_research/'+name)
 z.write(root/'sdxl_parallel/batching.py','sdxl_parallel/batching.py')
with zipfile.ZipFile(out/'sdxl_research_repack.zip') as z:assert z.testzip() is None

def table(head,rows):
 return '<table><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table>'
body=f'''<h1>每卡 32／16 实测与正式续训</h1><p>2026-10-23 · 172.18.33.60 · 物理卡 0、2 · A100 40GB</p>
<p class="good"><strong>当前默认方案下，每卡32两次测试均发生 OOM。每卡16完整通过，正式训练已从G2恢复，累积由8次降为2次，全局batch仍为64。</strong></p>
<h2>实际结果</h2>'''
body+=table(['每卡 batch','累积','结果'],[
 (4,8,'原正式训练 G1/G2 正常，约407–412秒/完整cycle'),
 (32,1,'第一次在fake反向OOM：张量约32.80GiB，预留未用约4.67GiB，申请1.25GiB失败'),
 (32,1,'重试通过5次fake/D，但在PCM反向OOM：张量约38.34GiB，申请320MiB失败'),
 (16,2,'4次G＋20次fake/D全部完成，含末步GAN与R1；随后从原G2续训'),
])
body+='''<p>重试曾尝试官方文档中的 <a href="https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf">expandable_segments</a> 设置，但实际 PyTorch 2.5.1 构建明确警告“不支持此平台”。它没有被视为已生效优化。第二次 OOM 时实际张量已接近容量上限；目前不把每卡32用于正式任务。每卡16运行未设置该选项。</p>
<h2>每卡16的显存和速度</h2>'''
body+=table(['指标','实测'],[
 ('单卡张量 allocated 峰值',f"{s['peak_allocated_GiB']:.2f} GiB"),
 ('单卡 reserved 峰值',f"{s['peak_reserved_GiB']:.2f} GiB"),
 ('nvidia-smi 采样最低空闲',f"{min(x['min_free_MiB'] for x in good['device_peaks'].values())/1024:.2f} GiB/卡"),
 ('预热后3轮平均耗时，含5次fake/D',f"{good['mean_timed_cycle_s']:.2f} 秒/完整cycle"),
 ('同为G2／阶段3：原每卡4',f"{old[1]['cycle_seconds']:.2f} 秒"),
 ('同为G2／阶段3：每卡16',f"{new[1]['cycle_seconds']:.2f} 秒"),
 ('这对G2测量的吞吐比',f"{ratio:.2f} 倍"),
])
body+='<p>这是少量更新的实测比较，未进行多次交替重复；不保证整段训练始终维持相同比例。两组全局batch、样本流、学习率、损失和5:1更新比保持一致，改变的是分包大小。浮点内核及归约顺序不同，不保证逐位相等。</p><h2>正式训练状态</h2>'
resumed_g=[x for x in resume['rows_at_capture']['0'] if x['phase']=='student_done']
if resumed_g:
 last=resumed_g[-1]
 body+=f"<p>正式恢复后已完成 G{last['g']}（阶段{last['stage']}），该轮耗时 {last['cycle_seconds']:.2f} 秒。恢复期间已通过 G3 末步GAN以及后续R1，记录的损失和梯度数值正常。恢复后的张量峰值 {last['peak_allocated_GiB']:.2f} GiB，缓存预留峰值 {last['peak_reserved_GiB']:.2f} GiB；后者高于独立短测，反映了不同的分配／缓存历史，不能用短测35.56GiB代替当前运行的峰值。</p>"
body+=table(['项目','当前设置'],[
 ('恢复位置','原 G2 / fake10 / D10，模型、优化器与随机状态一并加载'),
 ('训练方案','999/749/499/249；joint4末步GAN+DMD；完整10K训练集'),
 ('批量','2卡 × 每卡16 × 累积2 = 64；造样本batch16；VAE batch4'),
 ('目标与存档','继续至总G160；G40/80/120/160存权重，最后保存完整续训状态'),
 ('本次快照状态',f"{resume['status_at_capture']['status']}；G={resume['status_at_capture']['student_updates']}；fake={resume['status_at_capture']['fake_updates']}"),
 ('运行目录',resume['run_directory']),
])
body+='''<p>加入显式 --resume-repack，只放行 micro_batch 和 rollout_batch 的变化；仍要求全局batch、world size、模型及算法设置一致。48项CPU测试通过，实际两rank均已打印RESUMED，并从fake11继续更新，未从头重新训练正式模型。测试使用的新模型权重未用于续训。</p>
<h2>资源及验证边界</h2><ul><li>先安全停止原训练并完整保存G2状态，待两卡释放后逐次测试；三次测试没有并行叠放。</li><li>监测中未发现他人进程进入指定两卡，没有操作1、3号卡或他人的进程。正式续训保留同样的监测。</li><li>每卡16四轮测试覆盖学生2/3/4阶段、GAN、R1；损失和梯度范数均有限。</li><li>短测通过不能保证长时间无故障。样图质量、收敛和FID/Patch-FID不是这次batch测试的判断对象。</li></ul>
<p><a href="evidence/results.json">测试数据</a> · <a href="evidence/mb32.log">32首次失败日志</a> · <a href="evidence/mb32_retry.log">32重试日志</a> · <a href="evidence/mb16.log">16通过日志</a> · <a href="evidence/resume_record.json">实际续训记录</a> · <a href="evidence/cpu_tests.log">48项CPU测试</a> · <a href="sdxl_research_repack.zip">当前代码包</a></p>'''
doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>单卡batch调优与续训</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.75 system-ui,sans-serif;background:#f8fafc;color:#24354b}h1{font-size:30px}h2{font-size:23px;color:#195777;margin-top:30px}a{color:#1766aa}table{border-collapse:collapse;width:100%;background:white;margin:15px 0}th,td{border:1px solid #d6deea;padding:11px;text-align:left;overflow-wrap:anywhere}.good{padding:16px;background:#e5f3e9;border-left:5px solid #358253}</style>'+body+'</html>'
(out/'report.html').write_text(doc,encoding='utf-8')
print(json.dumps({'mean_cycle_s':good['mean_timed_cycle_s'],'matched_G2_speed_ratio':ratio,
 'allocated_GiB':s['peak_allocated_GiB'],'reserved_GiB':s['peak_reserved_GiB'],'resumed':True},indent=2))
