import hashlib, html, json, math, statistics
from pathlib import Path

root=Path('output/sdxl-crossdistill/turning_ablation_20261027/no_offload_4090')
checks={}; results={}; rows=[]
for arm,label in [('normalized_u','归一化转向'),('raw_angle','原始转角')]:
 p=root/arm
 b=json.loads((p/'benchmark.json').read_text())
 c=json.loads((p/'config.json').read_text())
 assert b['status']=='complete' and b['student_updates']==3 and b['fake_updates']==6 and b['D_updates']==0
 assert all(c[k] is False for k in ['fake_offload','offload_student','offload_student_optimizer'])
 ranks=[]
 for rank in [0,1]:
  logs=[json.loads(s) for s in (p/f'rank{rank}.jsonl').read_text().splitlines()]
  g=[r for r in logs if r['phase']=='student_done'];f=[r for r in logs if r['phase']=='fake']
  ts=[r for r in logs if r['phase']=='boundary_turning_sample']
  assert len(g)==3 and len(f)==6 and len(ts)==24
  for r in g:
   assert all(math.isfinite(r[k]) for k in ['grad_norm','dmd_loss','pcm_loss','middle_dm_loss','terminal_dm_loss'])
  assert all(math.isfinite(r['fake_grad_norm']) and math.isfinite(r['fake_loss']) for r in f)
  for r in ts:
   lo,hi={2:(509,739),3:(259,489),4:(20,239)}[r['stage']]
   assert all(lo<=t<=hi for t in r['timesteps'])
  ranks.append({'rank':rank,'student_updates':len(g),'fake_updates':len(f),'student_grad_norm':[r['grad_norm'] for r in g],
                'fake_grad_norm':[r['fake_grad_norm'] for r in f],'all_losses_finite':True,'timesteps_within_boundary':True})
 hashes=json.loads((p/'source_sha256.json').read_text())
 assert all(hashlib.sha256((Path('sdxl_research')/name).read_bytes()).hexdigest()==digest for name,digest in hashes.items() if name.endswith('.py'))
 checks[arm]={'no_offload':True,'source_matches_local':True,'ranks':ranks}
 results[arm]=b
 seconds=statistics.mean(b['cycle_seconds'][1:])
 rows.append(f'<tr><td>{label}</td><td>3 / 6</td><td>{seconds:.1f} 秒</td><td>{b["peak_allocated_GiB"]:.2f} GiB</td><td>{b["peak_reserved_GiB"]:.2f} GiB</td><td>{b["student_samples_per_second"]:.4f}</td></tr>')

a=json.loads((root/'normalized_u/config.json').read_text());b=json.loads((root/'raw_angle/config.json').read_text())
changed={k for k in a.keys()|b.keys() if a.get(k)!=b.get(k)}
checks['config_differences']=sorted(changed)
# Output/run paths and prior metadata may differ; training settings may not.
unexpected=changed-{'run','turning_profile','turning_sampling'}
assert not unexpected,unexpected
for rank in [0,1]:
 logs=[]
 for arm in ['normalized_u','raw_angle']:
  logs.append([json.loads(s) for s in (root/arm/f'rank{rank}.jsonl').read_text().splitlines()])
 ids=[[r['original_indices_rank'] for r in log if r['phase'] in ('fake','student_done')] for log in logs]
 assert ids[0]==ids[1]
checks['matched_sample_order']=True
(root/'validation.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
report='''<!doctype html><meta charset="utf-8"><title>双卡无卸载短测</title>
<style>body{font:16px/1.7 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#1e293b}table{border-collapse:collapse;width:100%}th,td{padding:12px;border-bottom:1px solid #cbd5e1;text-align:left}small{color:#64748b}code{background:#f1f5f9}</style>
<h1>两种采样加强：双卡无 CPU 卸载短测</h1>
<p>2026-10-27｜172.18.132.41｜GPU 4、7：RTX 4090 48GB。A100 没有空闲双卡，按已有备用服务器授权使用此服务器。</p>
<p><b>两组均完成 3 次学生更新、6 次 fake 更新，无 GAN。没有启动正式 G160。</b>每组从原始 SDXL 重新初始化，原有 10K 缓存，999/749/499/249 四点；学生每轮 PCM＋随机中段边界 DM＋末步 DM，教师 CFG 8。</p>
<table><tr><th>采样先验</th><th>学生 / fake 更新</th><th>完整周期均值</th><th>峰值实际分配/卡</th><th>峰值缓存预留/卡</th><th>学生样本/秒</th></tr>'''+''.join(rows)+'''</table>
<p>有效 batch 8＝2 卡×microbatch 1×累积 4；rollout batch 2。完整周期包含 2 次 fake 更新和 1 次学生更新，丢弃第一个周期，均值仅来自后两个周期。模型加载及评估存档不计入周期耗时。</p>
<p>学生、fake 和优化器均未卸载到 CPU。采用 FSDP 参数分片，学生学习率 1e-6、fake 5e-6。两组只改变学生 DM 重加噪时间步的采样先验，其他训练设置及样本顺序一致。</p>
<p>核验：两 rank 更新计数一致；全部已记录损失和梯度有限；所有采样时间在对应边界范围内；运行代码哈希与本地一致；未发生 OOM 或监测到选定卡进入其他计算任务。测试结束释放两卡。</p>
<p><b>解释边界：</b>这是执行与资源短测，不是采样先验的质量比较；没有保存模型或评测图像。它也不等同于正式 batch 64 的速度或长程稳定性验证。48GB 4090 的结果不能直接外推到 40GB A100。</p>
<p><a href="validation.json">逐 rank 核验记录</a> · <a href="normalized_u/benchmark.json">归一化组原始计时</a> · <a href="raw_angle/benchmark.json">原始转角组原始计时</a> · <a href="workflow.jsonl">运行记录</a></p>'''
(root/'report.html').write_text(report,encoding='utf-8')
print(json.dumps({'validation':'passed','config_differences':sorted(changed),'report':str(root/'report.html')},ensure_ascii=False))
