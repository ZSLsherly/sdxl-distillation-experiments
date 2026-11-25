import json
from pathlib import Path
root=Path('output/sdxl-crossdistill/turning_ablation_20261027')
status={'status':'code_deployed_GPU_short_test_incomplete','formal_training_started':False,
 'framework':'boundary DM: PCM + random middle 2/3 + terminal4; fake updates twice; GAN disabled',
 'CPU_tests_passed':20,'tiny_autograd_per_arm':{'G':4,'fake':8},
 'A100_attempt':{'student_updates':0,'fake_updates':2,'stop_reason':'CPU memory reserve; available 15.3 GiB'},
 '3090_attempt':{'student_updates':0,'fake_updates':1,'first_fake_update_seconds':632.757876,
   'stop_reason':'20 minute short-test wall limit','raw_angle_started':False},
 'nccl_scalar_check':{'passed':True,'seconds':.8616},
 'optimizer_probe':'optimizer_probe_cpu.json',
 'conclusion':'No comparable GPU training results yet. Keep both objectives unchanged; use a suitable training resource layout before G160.',
 'next_resource_layout':'Prefer two idle 40GB-class GPUs; student parameters and Adam on GPU, fake offload retained. Requires a fresh GPU short test.'}
(root/'execution_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
p=root/'protocol.json';protocol=json.loads(p.read_text(encoding='utf-8'));protocol['execution_status']=status
p.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
report=root/'report.html';s=report.read_text(encoding='utf-8')
s=s.replace('<b>状态：代码与分布准备，尚未训练。</b>','<b>状态：两组代码已部署，CPU验证通过；真实GPU短测未完成，G160尚未启动。</b>')
section='''<h2>实际执行记录：尚无可比较的训练结果</h2><table><tr><th>环境</th><th>完成fake更新</th><th>完成学生更新</th><th>停止原因</th></tr>
<tr><td>A100 0号卡，14GiB分配上限</td><td>2</td><td>0</td><td>主机可用内存降至15.3GiB，触发20GiB余量保护</td></tr>
<tr><td>3090 0号卡，允许短时让出计算</td><td>1</td><td>0</td><td>首个fake更新632.8秒，达到20分钟短测时限；原始转角组尚未启动</td></tr></table>
<p>两组各自通过4G/8fake的微型CPU反向传播检查，另有20项CPU/静态测试通过。它们不能替代真实SDXL/FSDP验证。GPU标量通信自检通过（约0.86秒）；目前异常耗时集中在首个fake反向计算之后的CPU梯度处理/Adam阶段，尚未拆分这两部分的完整模型耗时。</p>
<p>小尺寸CPU DTensor实验中foreach数值一致，但耗时波动明显，未据此启用优化器改动，也不将其宣称为确定加速。所有本方GPU测试进程均已退出。没有启动G160，没有生成两组的质量评分。</p>
<p>优先等待两张空闲40GB级GPU，再验证student参数和Adam驻留GPU、fake继续卸载的运行布局；这是运行资源调整，训练目标和两组采样定义保持不变。该双卡布局目前只有配置，尚未实测。</p>
<p><a href="execution_status.json">执行状态</a> · <a href="boundary_cpu_smoke.json">CPU反向传播</a> · <a href="a100_probe_stopped.json">A100停止记录</a> · <a href="gpu_probe_records/workflow/retry1_summary.json">3090停止记录</a> · <a href="optimizer_probe_cpu.json">小尺寸CPU优化器诊断</a></p>'''
s=s.replace('<h2>实际训练使用的三个条件分布</h2>',section+'<h2>实际训练使用的三个条件分布</h2>')
report.write_text(s,encoding='utf-8')
for arm in ['normalized_u','raw_angle']:
 p=Path(f'sdxl_research/configs/turning_{arm}_g160.json');cfg=json.loads(p.read_text())
 cfg.update(offload_student=False,micro_batch=8,rollout_batch=8)
 Path(f'sdxl_research/configs/turning_{arm}_g160_2gpu_candidate.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
print(json.dumps(status,indent=2))
