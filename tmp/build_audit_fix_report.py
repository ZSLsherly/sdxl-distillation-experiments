import hashlib
import json
import re
from pathlib import Path
import shutil
import zipfile

root=Path(__file__).resolve().parents[1]
src=root/'sdxl_research'
out=root/'output/sdxl-crossdistill/audit_fixes_20261022'
evidence=out/'evidence'
evidence.mkdir(parents=True,exist_ok=True)
reference=root/'tmp/crossdistill_upstream_20261022.py'
shutil.copyfile(reference,evidence/'crossdistill_upstream.py')
manifest=json.loads((src/'source_manifest.json').read_text(encoding='utf-8'))
assert all(hashlib.sha256((src/p).read_bytes()).hexdigest()==h for p,h in manifest.items())
log=(evidence/'cpu_tests.log').read_text()
assert 'Ran 38 tests' in log and log.rstrip().endswith('OK')
record={
 'date':'2026-10-22', 'tests':38, 'result':'PASS', 'torch':'2.5.1+cu121',
 'CUDA_VISIBLE_DEVICES':'', 'visible_GPUs':0, 'priority':'nice 19; OMP/MKL/OPENBLAS 2 threads',
 'remote_validation_directory':'/data5/shenglin.zeng/sdxl_crossdistill/validation/audit_fixes_20261022',
 'gpu_training_started':False,
 'source_sha256':manifest,
 'reference':{'url':'https://raw.githubusercontent.com/AlibabaResearch/SparkDiffusion/main/sparkdiffusion/models/t2v_model_distill.py',
              'sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
              'student_lines':[1019,1028],'critic_lines':[1300,1309]},
 'limits':['CPU checks use small tensors/networks, not a full pretrained SDXL training run',
           'No new multi-GPU numerical equivalence or throughput claim',
           'CrossDistill uses a shared mixture score, not one exact stage-conditioned score per stage'],
}
(evidence/'validation.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(out/'sdxl_research_fixed.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in list(manifest)+['source_manifest.json']:
  z.write(src/name,'sdxl_research/'+name)
 z.write(root/'sdxl_parallel/batching.py','sdxl_parallel/batching.py')
with zipfile.ZipFile(out/'sdxl_research_fixed.zip') as z:
 assert z.testzip() is None
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SDXL 三项审计修复与验证</title>
<style>body{max-width:1060px;margin:40px auto;padding:0 25px;font:17px/1.75 system-ui,sans-serif;background:#f8fafc;color:#243247}h1{font-size:30px}h2{font-size:23px;color:#145176;margin-top:30px}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d4dde7;padding:12px;text-align:left}a{color:#1566a3}code{background:#e9eef4;padding:2px 5px}li{margin:6px 0}.status{background:#e1f2e9;border-left:5px solid #327c53;padding:16px}p,td{overflow-wrap:anywhere}</style>
<h1>三项审计问题已修复</h1><p>2026-10-22 · 静态优先 · 38 项 CPU 回归测试通过</p>
<p class="status">本轮修改的是 sdxl_research，服务器只上传至独立验证目录。CUDA 完全禁用，未启动 GPU 训练，未替换已有实验代码或权重。完整预训练 SDXL 多卡短训练尚未验证。</p>
<h2>1 · 按 CrossDistill 的共享 fake 方案补齐阶段覆盖</h2>
<p>重新下载并核对了<a href="https://github.com/AlibabaResearch/SparkDiffusion/blob/main/sparkdiffusion/models/t2v_model_distill.py#L1300">官方训练源码</a>，并保存本次下载内容及 SHA256。非 joint 分支中，critic 按更新计数轮流使用 1、2、3、4 步输出；开启 diversity 监督时，学生按更新计数轮流使用 2、3、4 步。我们采用相同的阶段轮转关系，保留 SDXL 的 VP/DDIM 参数化。</p>
<table><tr><th>部分</th><th>修复后</th></tr><tr><td>fake-score 训练样本</td><td>1 → 2 → 3 → 4 → 1…，由已完成 fake 更新计数决定</td></tr><tr><td>学生更新</td><td>继续 2 → 3 → 4，PCM、DMD/GAN 权重及 5:1 比例不变</td></tr><tr><td>GAN 判别器负样本</td><td>仍为四步最终图，不随 fake-score 阶段改变</td></tr><tr><td>造样本开销</td><td>GAN 组一条四步轨迹，同时捕获所选阶段 clean 输出与最终图；不重复执行前缀 UNet。无 GAN 组只运行所需步数。</td></tr><tr><td>续训</td><td>增加 fake_stage_policy 标识，拒绝把旧“仅四步 fake”完整状态静默接入新协议。新协议恢复按 fcount 继续阶段轮转。</td></tr></table>
<p>这里沿用的是<strong>共享 fake 学习混合输出分布</strong>的做法，不宣称每个阶段都有独立且精确的条件 score。没有增加额外学生或 fake 网络。由于训练目标分布修正了，新实验应从原始 SDXL 开始，和旧协议结果分开标注。</p>
<h2>2 · CPU 卸载参数冲突</h2><p>同时设置两种 student offload 时，在加载模型前直接报清楚的参数错误。只开启其中一种的已有路径保持原样，无需改造优化器。</p>
<h2>3 · 评测特征缓存与 worker 数解耦</h2><p>先按全局样本索引固定每块 64 张，再把块分给 worker。新格式下改变 worker 数不会改变缓存文件名。旧版按 rank 切割的缓存如含非标准边界，会明确提示使用新评测目录；不自动删除或重写历史产物。</p>
<h2>本轮小规模验证</h2><ul>
<li>38 项 CPU 测试全部通过，包含新增 5 项审计回归检查。</li>
<li>5 个小张量样本以 2/2/1 分块，对全部四个阶段比较：捕获的中间 clean 输出、四步最终输出，与分别执行对应步数结果逐元素完全一致；fake 小型优化器更新也完全一致。</li>
<li>每块只有四次预测调用，验证 GAN 组未重复执行前缀；两路造样本输出均不带梯度图。</li>
<li>800 次 fake 更新中，每个阶段各 200 次；在中途拆分计数后，轮转序列仍与连续执行一致。旧协议完整状态会被拒绝。</li>
<li>COCO 1K/10K 模拟从 2 个 worker 改为 1、3、4、2 个，最终索引完整且无重复。</li>
<li>已有 batch 梯度、R1、冻结 VAE/DINO 输入梯度、SDXL 条件接口、CFG 与评测采样点等回归检查继续通过。</li>
<li>首轮新增测试捕获了双输出拼接的 detach 错误；已修复，以上结果来自重新完整运行的最终版本。</li></ul>
<p>实际验证环境：PyTorch 2.5.1+cu121，CUDA_VISIBLE_DEVICES 为空，可见 GPU 数为 0。采用 nice 19 和低线程数，最终源码哈希 42 项核验一致。CPU 测试通过不等于全量 SDXL 多卡稳定性或生成质量已得到证明。</p>
<h2>资源与交付</h2><p>VPN 重连后只读检查显示，172.18.33.60 四张 A100 均有任务，利用率约 63%–99%。依照“静态优先”和不影响他人训练的要求，本轮没有进行 GPU 短测。</p>
<p><a href="sdxl_research_fixed.zip">修复后代码包</a> · <a href="evidence/cpu_tests.log">38 项测试日志</a> · <a href="evidence/validation.json">源码哈希与验证记录</a> · <a href="evidence/crossdistill_upstream.py">本次官方源码快照</a></p>
</html>'''
(out/'report.html').write_text(html,encoding='utf-8')
for target in re.findall(r'href="([^"]+)"', html):
 if not target.startswith('http'):
  assert (out/target).exists(), target
old=root/'output/sdxl-crossdistill/independent_audit_20261022/report.html'
text=old.read_text(encoding='utf-8')
banner='<p style="padding:15px;background:#e1f2e9"><strong>修复更新：</strong>本报告中的三项问题已处理，<a href="../audit_fixes_20261022/report.html">查看修复与 38 项 CPU 验证结果</a>。下文保留为原审计记录。</p>'
if '../audit_fixes_20261022/report.html' not in text:
 text=text.replace('<h1>',banner+'<h1>',1);old.write_text(text,encoding='utf-8')
print(out/'report.html')
