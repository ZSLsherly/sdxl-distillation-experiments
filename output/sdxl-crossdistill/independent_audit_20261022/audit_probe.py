"""CPU/stdlib-only audit probes. Does not import training code or touch GPUs."""
import argparse
import ast
from collections import Counter
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'sdxl_research/train.py').exists())
SRC = ROOT / 'sdxl_research'
OUT = ROOT / 'output/sdxl-crossdistill/independent_audit_20261022'
OUT.mkdir(parents=True, exist_ok=True)
tree = ast.parse((SRC / 'train.py').read_text(encoding='utf-8'))
funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
runtime = ast.parse((SRC / 'runtime.py').read_text(encoding='utf-8'))
schedule = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == 'schedule_settings')
namespace = {'argparse': argparse, 'Path': Path, 'os': os}
exec(compile(ast.Module(body=[schedule, funcs['arguments']], type_ignores=[]), str(SRC/'train.py'), 'exec'), namespace)
old_argv = sys.argv
try:
    sys.argv = ['train.py', '--run', 'audit-only', '--variant', 'joint4', '--offload-student', '--offload-student-optimizer']
    args = namespace['arguments']()
finally:
    sys.argv = old_argv
assert args.offload_student and args.offload_student_optimizer

rollout = funcs['rollout']
defaults = {a.arg: ast.literal_eval(v) for a, v in zip(rollout.args.args[-len(rollout.args.defaults):], rollout.args.defaults)}
assert defaults['stage'] == 4
fake_call = next(n for n in ast.walk(funcs['generate_fake_batch']) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'rollout_chunks')
assert len(fake_call.args) == 4 and not fake_call.keywords
batch_tree = ast.parse((SRC/'batching.py').read_text(encoding='utf-8'))
chunk_fn = next(n for n in batch_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'rollout_chunks')
inner_call = next(n for n in ast.walk(chunk_fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'rollout')
assert len(inner_call.args) == 2 and not inner_call.keywords
student_stages = [2 + g % 3 for g in range(160)]

def chunks(n, rank, world):
    start, end = n*rank//world, n*(rank+1)//world
    return [(left, min(left+64, end)) for left in range(start, end, 64)]
files = set(chunks(10000, 0, 2)) | set(chunks(10000, 0, 1))
counts = Counter(i for left, right in files for i in range(left, right))
duplicates = sorted(i for i, count in counts.items() if count > 1)
assert sum(counts.values()) == 10008 and duplicates == list(range(4992, 5000))

manifest = json.loads((SRC/'source_manifest.json').read_text(encoding='utf-8'))
hashes = {name: hashlib.sha256((SRC/name).read_bytes()).hexdigest() for name in manifest}
evidence = {
    'scope': 'Local source review; no remote access, no GPU run, no training code edits',
    'source_manifest_mismatches': [k for k in manifest if manifest[k] != hashes[k]],
    'source_sha256': hashes,
    'fake_rollout_default_stage': defaults['stage'],
    'fake_calls_omit_stage': True,
    'student_stage_counts_at_G160': dict(Counter(student_stages)),
    'double_offload_arguments_accepted': True,
    'double_offload_GPU_runtime_reproduced': False,
    'evaluation_world_2_rank0_then_world_1': {
        'rows': sum(counts.values()), 'unique_rows': len(counts), 'duplicates': duplicates,
        'overlapping_files': ['04992_05000.npz', '04992_05056.npz']},
    'local_torch_tests_rerun': False,
}
(OUT/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')

def link(file, line):
    return f'<a href="{(SRC/file).as_uri()}#L{line}">{html.escape(file)}:{line}</a>'

body = f'''<h1>SDXL 独立代码审计</h1>
<p class="muted">2026-10-22 · 当前 sdxl_research · 三个独立审阅分工，主审逐项复核</p>
<p><strong>结论：发现 1 项高优先级算法偏差、2 项条件触发的运行问题。建议先处理第 1 项，再开展正式扩训。</strong>本次没有修改训练代码，没有连接服务器或启动 GPU。已有短测能跑通，不代表目标函数正确或完整训练已验证。</p>
<h2>P1 · fake-score 的训练分布与第 2、3 步 DMD 输入不一致</h2>
<p>{link('train.py',292)} 调用 rollout_chunks，后者 {link('batching.py',120)} 不传 stage，使用 {link('train.py',263)} 的默认 stage=4。因此 fake 网络始终用四步最终 x0 加噪训练。学生却在 {link('train.py',444)} 轮流选择 2、3、4 步，{link('train.py',525)} 从所选步直接预测 clean x0，再于 {link('train.py',532)} 查询同一个 fake。</p>
<p>用 q₂、q₃、q₄ 表示三个提前终止输出分布：fake 学的是 q₄ 的加噪分布，而第 2、3 步更新需要对应目标定义下的 score。代码没有保证 q₂=q₃=q₄，也没有 stage 条件。因此这些更新不能直接解释为各自生成分布上的标准 DMD。G160 中第 2、3 步分别执行 54、53 次，影响范围不小。</p>
<p>这是从 sdxl_parallel 继承的问题，非本次优化新增。它不会改变已生成图像的 FID 计算，但会限制“规范 DMD 对照”和分阶段机制的解释。不能凭此推断 GAN 的已有收益全部无效。</p>
<p><strong>最小处理方向：</strong>先明确训练目标采用哪个输出分布，再让 fake 的造样本策略与之对应。可先试同一训练 cycle 内 fake 与 G 使用同一 stage；共享 fake 会跨阶段追踪，因此仍需验证拟合情况。仅将 fake 改为混合 stage，也不能声称获得严格的逐 stage 条件 score。不要为此立即增加三套大网络。</p>
<h2>P2 · 两种学生卸载同时开启时，优化器设备冲突</h2>
<p>参数解析允许 --offload-student 与 --offload-student-optimizer 同开。前者在 {link('train.py',172)} 让 FSDP 参数和梯度卸载到 CPU；后者在 {link('train.py',574)} 每次更新前无条件把 Adam moments 移到 CUDA。第一次更新创建状态后，后续更新将出现 CPU 参数/梯度与 CUDA moments 的不匹配。</p>
<p>标准库探针已确认解析器接受此组合；设备冲突依据代码路径确认，本轮没有 GPU 复现。当前配置未同时开启两项，因此不影响已完成短测。<strong>最小处理：</strong>拒绝该组合，或按参数实际设备迁移状态；不需要重写优化器。</p>
<h2>P2 · 评测续跑更改 worker 数会产生重叠特征块</h2>
<p>{link('evaluation_core.py',194)} 先按 rank/world 切样本，再在每段内每 64 张保存。COCO10K 先完成 world=2 的 rank0，之后用 world=1 续跑，会同时留下 04992_05000.npz 与 04992_05056.npz。</p>
<p>CPU 索引复现结果：<strong>10,008 行、10,000 个唯一索引，4992–4999 重复</strong>。{link('evaluation_core.py',262)} 会拒绝出分，所以不会静默产生错误 FID，但会浪费已完成的生成工作。固定 world 续跑不受影响。</p>
<p><strong>最小处理：</strong>先固定全局 64 张分块，再按块分配 worker；或者暂时保存并强制相同 world。</p>
<h2>已核对与保留的边界</h2>
<ul><li>没有发现当前 loss/accumulation 再经 FSDP 均值的直接重复缩放错误；GAN 经 VAE/DINO 回到学生输入的梯度路径没有明显 detach。</li>
<li>停止标志由 rank0 广播，完整存档写新目录；未发现旧版独立计时造成的同类不同步问题。</li>
<li>自定义评测读取 checkpoint 的 schedule；FID、Patch-FID 处理与本地保留的 DMD2 参考实现一致。</li>
<li>本轮未重跑既有 33 项 Torch 测试：本地运行时没有 PyTorch。历史测试日志与 GPU 短测不是本轮重新验证。</li>
<li>局部 VAE 解码与整图裁剪不等价、中心 Patch-FID 无法代表所有纹理、少量样本校准的 GAN 系数未保证长期梯度比例，仍是实验局限。</li>
<li>可考虑启动时使用已检查 GPU 的 UUID，消除设备编号解释的歧义；本轮没有设备映射实证，不把它列为已确认跑错卡。启动检查也不等于独占预约。</li></ul>
<p>不建议此时增加框架或大规模重构。先明确第 1 项目标分布，做少量拟合与梯度检查，再处理两处小范围运行问题即可。</p>
<p><a href="evidence.json">本轮 CPU/AST 探针与源码哈希</a> · <a href="audit_probe.py">可复跑探针</a></p>'''
document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SDXL 独立代码审计</title><style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:17px/1.75 system-ui,sans-serif;color:#223047;background:#f8fafc}h1{font-size:30px}h2{font-size:22px;margin-top:32px;color:#163b65}a{color:#155da3}.muted{color:#64748b}p,li{overflow-wrap:anywhere}strong{color:#8b361d}</style>'+body+'</html>'
(OUT/'report.html').write_text(document, encoding='utf-8')
(OUT/'audit_probe.py').write_text(Path(__file__).read_text(encoding='utf-8'), encoding='utf-8')
print(json.dumps({k:v for k,v in evidence.items() if k!='source_sha256'}, ensure_ascii=False, indent=2))
