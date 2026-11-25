from pathlib import Path
import sys,json,hashlib,zipfile,html
root=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(root/'.plot_runtime_pkgs'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=root/'output/sdxl-crossdistill/small_test_20261022'
data=json.loads((out/'evidence/results.json').read_text(encoding='utf-8'))
names=['legacy_b8','legacy_offload_b8','block_offload_b8']
labels=['Fine / GPU optimizer','Fine / CPU optimizer','Block + bucket / CPU optimizer']
rows=[data['runs'][name] for name in names]
plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(11.5,4.6))
for ax,values,title in [(axes[0],[r['mean_cycle_seconds'] for r in rows],'Seconds / student update'),
                       (axes[1],[r['status']['peak_reserved_GiB'] for r in rows],'Peak reserved GiB / GPU')]:
 ax.barh(labels,values,color=['#97a3af','#2e7f93','#cf9e66'])
 for i,v in enumerate(values):ax.text(v+.4,i,f'{v:.2f}',va='center')
 ax.set_xlim(0,max(values)*1.15);ax.set_title(title)
fig.suptitle('Actual SDXL: 2 A100, batch8, 1024px, fake:G = 5:1\nOne run/config; first update warmup, next 3 timed; interrupted run excluded',fontsize=12)
fig.tight_layout();fig.savefig(out/'results.png');fig.savefig(out/'results.pdf');plt.close(fig)
def table(headers,rows):
 return '<table><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</table>'
slow=(rows[2]['mean_cycle_seconds']/rows[1]['mean_cycle_seconds']-1)*100
off=(rows[1]['mean_cycle_seconds']/rows[0]['mean_cycle_seconds']-1)*100
body='''<h1>SDXL 小规模实测与代码复核</h1><p class="meta">2026-10-22 · 172.18.33.60 · A100 0、3号卡 · 所有本次GPU进程已退出</p>
<aside><b>结论：</b>三个真实SDXL短测完成，GAN与R1分支正常；显存压力通过现有CPU优化器卸载得到缓解。粗粒度FSDP没有加速，默认不启用。最后一组被中途进入0号卡的其他进程打断，不能作为性能证据。</aside>
<h2>1. 实测设置与结果</h2><p>使用真实SDXL、DINO/RGB判别器、FP32 VAE和原10K数据；1024px，总batch8，两卡各micro4。每组4次学生更新、20次fake/D更新，阶段2/3/4/2；第3次G使用末步GAN，第16次D使用R1。普通四点999/749/499/249，PCM边界749，DMD再加噪20–919。未降低分辨率、未把5:1改成1:1。</p>
<p>计时剔除首个G更新，统计剩余3个周期；包含fake/D、G与PCM，不含模型初始化、评估和完整SDXL存档。每配置仅一次短测，且服务器其他卡有任务，不能当作稳定吞吐或batch64结果。</p>'''
body+=table(['版本','秒/G ↓','allocated峰值','reserved峰值','结果'],[
 ['原细粒度，学生优化器在GPU',f"{rows[0]['mean_cycle_seconds']:.2f}",'29.78GiB','38.45GiB','完成，但40GB卡余量过小'],
 ['原细粒度，学生优化器状态在CPU',f"{rows[1]['mean_cycle_seconds']:.2f}",'24.78GiB','33.17GiB','完成；作为低显存基线'],
 ['粗粒度＋分桶＋周期控制，同样CPU卸载',f"{rows[2]['mean_cycle_seconds']:.2f}",'24.75GiB','33.42GiB',f'完成，比匹配基线慢{slow:.1f}%'],
 ['细粒度＋分桶＋周期控制，同样CPU卸载','不计入','—','—','外部进程进入GPU0后，本次worker OOM并退出'],
])
body+='<img src="results.png" alt="三个已完成短测的速度和显存比较">'
body+=f'''<p>CPU卸载减少了约5.28GiB reserved峰值，代价是本轮每G慢约{off:.1f}%。粗粒度FSDP把非空分组从103减到33，判别器通信从每次28次变为1次，但没有转化成端到端提速。因此正式配置保留细粒度FSDP；分桶/周期控制仍是可选测试项。没有开启compile或foreach来叠加未经验证的变量。</p>
<h2>2. 正确性证据与边界</h2><ul><li>三个完成运行都覆盖GAN、R1和20次fake/D更新；两rank日志中的损失与梯度范数有限，没有OOM或NCCL错误。</li><li>各组按位置抽取的样本编号一致；但完整SDXL日志数值不是逐bit一致。粗粒度相对CPU卸载基线的DMD loss最大相对差约1.56%，G梯度范数约0.94%；仅改变优化器卸载的两组也有小幅差异。没有重复同配置A/A测试，不能把所有差异归因于某一个优化，也不能宣称完整梯度等价。</li><li>独立的两GPU小模型FSDP“保存→加载→继续更新”检查通过：参数和Adam状态与连续更新逐元素完全一致。这验证存档机制，不等同于完整SDXL大存档已测试。</li><li>代码修正后，禁用CUDA运行unittest自动发现：33项CPU测试全部通过。GPU/Gloo脚本加了main入口保护，导入时不再自动启动分布式或占GPU。</li><li>核对了实际安装的PyTorch2.5.1 FSDP2源码：CPU卸载梯度累积使用已有梯度加上新梯度，并在累积时等待D2H完成。没有把FSDP1的限制直接套到当前实现；正式batch64累积仍需实际验证。</li></ul>
<h2>3. 这次修正的代码问题</h2>'''
body+=table(['问题','修正','范围'],[
 ['--short可能保留benchmark=True','显式切回正常短训练模式，保证4次更新及评估/存档路径','launch.py；CPU回归覆盖'],
 ['默认数据目录还是早期gan_v3','改为当前parallel_10k_seed20261018','train.py；现有显式配置与已跑测试不变'],
 ['被提前停止的benchmark可能进入性能汇总','要求完成指定次数且测量窗口覆盖2/3/4阶段','summarize.py；失败组不输出速度结论'],
 ['end_to_end_seconds实际不含模型初始化','改名training_phase_seconds；launcher另记完整子进程wall time','避免低估实际实验耗时'],
 ['GPU/Gloo测试脚本导入即执行','移入main函数，仅直接运行时执行','unittest discover不会误启动GPU'],
 ['默认候选过多、未证实收益','默认性能对比只保留原路径与简单分桶/周期控制；科学配置默认原路径','block/compile/foreach保持关闭'],
])
body+='''<p>这些修正不引入新的训练框架、自动队列或显存管理器。GAN、DMD、PCM公式、5:1更新比例、teacher CFG、数据顺序和随机数流没有因性能优化被替换。旧sdxl_parallel和已完成实验权重保持原样。</p>
<h2>4. 中途占卡事件</h2><p>每组启动前都检查了0、3号卡没有计算进程，测试使用较低CPU/IO优先级。最后一组运行中，另一个用户的进程1899365进入GPU0，占约17GiB；日志记录本次worker分配显存失败。torchrun随后退出，复核时本次两个worker均已消失，没有对其他进程发送信号，也没有重置GPU或服务器。</p>
<p>17:24复核：本次8个完整SDXL worker均已退出；GPU0只显示外部进程。所有短测输出合计约1.2MB，没有产生大型SDXL权重存档。本轮不能声称共享机器完全零性能影响；启动时空闲不等于资源已预约，后续完整对照应先安排独占时段。</p>
<h2>5. 当前建议</h2><p>保留细粒度FSDP和原通信路径，优先使用已存在的CPU优化器卸载降低两卡内存压力。新提供的uniform_joint4_2gpu_memory配置仍保持batch64，但micro/rollout为4并开启优化器卸载；它只是保守候选，尚未完成batch64 GPU验证。</p>
<p>暂不继续扩展优化组合，也不自动开始160次更新。取得可靠空闲窗口后，只需补“同设置基线 vs 简单通信合并”的完整小测；若没有可重复收益，就保留原路径。不要凭这次3个计时点承诺稳定提速。</p>
<h2>文件与证据</h2><p><a href="evidence/results.json">全部指标、样本一致性及源代码哈希</a> · <a href="evidence/state_gpu.log">GPU恢复检查</a> · <a href="evidence/cpu_tests.log">33项CPU回归</a> · <a href="evidence/interrupted_run.log">中断日志</a> · <a href="evidence/release.txt">释放核查</a> · <a href="sdxl_research_reviewed.zip">最新代码包</a></p>
<p><a href="../optimization_plan_20261022/report.html">之前的优化设计与后续实验方案</a>（该页是实测前设计记录，以本页的实测结论和最新代码默认值为准）。</p>'''
style='''body{margin:0;background:#f3f6f8;color:#213342;font:16px/1.85 "Segoe UI","Microsoft YaHei",sans-serif}main{max-width:1080px;margin:35px auto;background:white;padding:32px}h1{font-size:30px;color:#15485c}h2{font-size:22px;margin-top:32px}.meta{color:#637484}aside{padding:18px;background:#e6f2f6;border-left:5px solid #2e7f93}table{width:100%;border-collapse:collapse;font-size:14px}td,th{text-align:left;vertical-align:top;padding:11px;border-bottom:1px solid #d9e1e7}th{background:#edf3f6}img{max-width:100%;height:auto}a{color:#176e88}li{margin:8px 0}@media(max-width:700px){main{padding:16px;margin:10px}table{display:block;overflow:auto}}'''
(out/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SDXL小规模实测与复核</title><style>'+style+'</style><main>'+body+'</main></html>',encoding='utf-8')
project=root/'sdxl_research'
manifest={p.relative_to(project).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in project.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='source_manifest.json'}
(project/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with zipfile.ZipFile(out/'sdxl_research_reviewed.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in project.rglob('*'):
  if p.is_file() and '__pycache__' not in p.parts:z.write(p,'sdxl_research/'+p.relative_to(project).as_posix())
 z.write(root/'sdxl_parallel/batching.py','sdxl_parallel/batching.py')
old=root/'output/sdxl-crossdistill/optimization_plan_20261022/report.html'
s=old.read_text(encoding='utf-8')
notice='<aside id="gpu-review-update"><b>实测更新：</b>本页保留实测前设计记录。GPU短测、修正后的默认配置和最新代码见 <a href="../small_test_20261022/report.html">小规模实测与代码复核报告</a>；粗粒度FSDP本次未提速，默认不启用。</aside>'
if 'id="gpu-review-update"' not in s:old.write_text(s.replace('<main>','<main>'+notice,1),encoding='utf-8')
print(out/'report.html')
