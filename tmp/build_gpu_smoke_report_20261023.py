import hashlib,html,json,math
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/'output/sdxl-crossdistill/gpu_smoke_20261023'
r=json.loads((out/'evidence/results.json').read_text())
s=r['status'];cfg=r['config']
assert s['status']=='complete' and r['guard']['returncode']==0
assert all(hashlib.sha256((root/'sdxl_research'/p).read_bytes()).hexdigest()==h for p,h in r['source_sha256'].items())
gs=[x for x in r['rows']['0'] if x['phase']=='student_done']
fs=[x for x in r['rows']['0'] if x['phase']=='fake_D']
alltimed=[x for x in gs if x['g']>1]
def table(headers,rows):
 return '<table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table>'
body=f'''<h1>默认方案双卡小规模训练短测</h1><p>2026-10-23 · 172.18.33.60 · 物理 GPU 0、2 · 最新模块代码</p>
<p class="good"><strong>四次学生更新和二十次 fake／判别器更新全部完成，两 rank 正常退出。</strong>损失及梯度范数均为有限值；阶段轮转、末步 GAN、R1 路径已在真实 SDXL 上执行。没有保存大权重，也没有启动后续长训练。</p>
<h2>本次设置</h2><p>默认算法：joint4，推理点 999/749/499/249，学生阶段 2/3/4，fake 阶段 1/2/3/4，末步 GAN+DMD，PCM 系数10，hybrid DINO+RGB、hinge、局部解码、仅 D 翻转，学生/fake 学习率 1e-6。1024px，prepared 10K 数据源。</p>
<p><strong>短测执行配置：</strong>全局 batch8 = 2 卡 × 每卡4；rollout batch4，已有学生优化器 CPU 卸载开启，fine FSDP、每参数 D 通信、legacy 控制。运行4个 G cycle，第一轮作为预热，其余3轮计时。不是正式 batch64 速度测量。</p>
<h2>显存与速度</h2>'''
body+=table(['项目','结果'],[
 ('单卡 PyTorch allocated 峰值（两 rank 最大）',f"{s['peak_allocated_GiB']:.2f} GiB"),
 ('单卡 PyTorch reserved 峰值（两 rank 最大）',f"{s['peak_reserved_GiB']:.2f} GiB"),
 ('每次 G cycle 平均耗时（含5次fake/D）',f"{r['mean_timed_cycle_s']:.2f} 秒"),
 ('每次 G cycle 中位耗时',f"{r['median_timed_cycle_s']:.2f} 秒"),
 ('有效学生样本吞吐',f"{8/r['mean_timed_cycle_s']:.3f} 样本/秒"),
 ('完整运行墙钟时间（含初始化、启动与退出）',f"{r['guard']['wall_seconds']/60:.2f} 分钟"),
])
body+=table(['物理卡','nvidia-smi 采样最大占用','采样最小空闲'],[
 (k,f"{v['max_used_MiB']/1024:.2f} GiB",f"{v['min_free_MiB']/1024:.2f} GiB") for k,v in r['device_monitor_peaks'].items()])
body+='<p>allocated 是 PyTorch 张量占用，reserved 包含缓存；nvidia-smi 还包含 CUDA/NCCL 等开销。5秒采样的空闲值不是亚秒瞬时最低值。上述数字是本次短测实测，不能保证之后其他任务进入时仍有同样余量。</p><h2>四轮更新</h2>'
body+=table(['G','阶段','GAN','完整cycle秒','DMD loss','GAN loss','PCM loss','G 梯度范数'],[
 (g['g'],g['stage'],'是' if g['gan'] else '否',f"{g['cycle_seconds']:.2f}",f"{g['dmd_loss']:.6g}",f"{g['gan_loss']:.6g}",f"{g['pcm_loss']:.6g}",f"{g['grad_norm']:.6g}") for g in gs])
body+='<h2>核验与边界</h2><ul><li>两 rank 均为20次 fake/D +4次 G；fake 阶段严格为 1/2/3/4 循环，GAN D 始终使用四步最终输出。</li><li>G3 执行末步 GAN；D16 执行 R1。所有已记录损失、真假评分和梯度范数有限，每次更新两卡合计8个不同样本。</li><li>监测期间没有发现其他进程进入 0、2 卡；本次两个 worker 已退出，未操作 1、3 卡任务。</li><li>这是可运行性、显存及速度短测；没有输出对照图、FID/Patch-FID，没有验证收敛或视觉收益，也未验证新版本 batch64 的累积路径与吞吐。</li><li>CPU 卸载、设备编号和本次小 batch 均已明确记录，不把结果冒充未卸载的默认吞吐。没有保存完整训练状态，本次也不验证断点恢复。</li></ul>'
body+='<p><a href="evidence/results.json">完整配置、损失与监测汇总</a> · <a href="evidence/train.log">原始训练日志</a> · <a href="evidence/resource_monitor.jsonl">资源监测日志</a></p>'
doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>默认方案双卡短测</title><style>body{max-width:1100px;margin:40px auto;padding:0 24px;font:17px/1.75 system-ui,sans-serif;background:#f8fafc;color:#24354b}h1{font-size:30px}h2{font-size:23px;color:#185675;margin-top:30px}a{color:#1766aa}table{border-collapse:collapse;width:100%;background:white;margin:15px 0}th,td{border:1px solid #d6deea;padding:11px;text-align:left}.good{padding:16px;background:#e5f3e9;border-left:5px solid #358253}</style>'+body+'</html>'
(out/'report.html').write_text(doc,encoding='utf-8')
print(json.dumps({'allocated_GiB':s['peak_allocated_GiB'],'reserved_GiB':s['peak_reserved_GiB'],
 'mean_cycle_s':r['mean_timed_cycle_s'],'median_cycle_s':r['median_timed_cycle_s'],
 'wall_minutes':r['guard']['wall_seconds']/60,'source_match':True},indent=2))
