"""Independent read-only source checks; no torch import, GPU or remote access."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
from types import SimpleNamespace
from unittest.mock import patch

root=Path(__file__).resolve().parents[1]
src=root/'sdxl_research'
out=root/'output/sdxl-crossdistill/independent_reaudit_20261022'
out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((src/'source_manifest.json').read_text(encoding='utf-8'))
hashes={name:hashlib.sha256((src/name).read_bytes()).hexdigest() for name in manifest}
assert hashes==manifest
prior=json.loads((root/'output/sdxl-crossdistill/audit_fixes_20261022/evidence/validation.json').read_text(encoding='utf-8'))
assert hashes==prior['source_sha256']
tree=ast.parse((src/'evaluation_core.py').read_text(encoding='utf-8'))
fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='feature_blocks')
namespace={}
exec(compile(ast.Module(body=[fn],type_ignores=[]),'feature_blocks','exec'),namespace)
blocks=namespace['feature_blocks']
checks=0
for n in (1,63,64,65,1000,10000):
 for old_world in (1,2,3,4,7):
  cached=set(blocks(n,0,old_world))
  for new_world in (1,2,3,4,7):
   assignments=[b for rank in range(new_world) for b in blocks(n,rank,new_world)]
   assert len(assignments)==len(set(assignments))
   completed=cached|set(assignments)
   ids=sorted(i for l,r in completed for i in range(l,r))
   assert ids==list(range(n))
   checks+=1

spec=importlib.util.spec_from_file_location('audited_launch',src/'launch.py')
launch=importlib.util.module_from_spec(spec);spec.loader.exec_module(launch)
with contextlib.nullcontext(str(out)) as temp:
 config=json.loads((src/'configs/uniform_joint4.json').read_text())
 config['resume']='/example/same-protocol/training_state/g000040'
 cfg=Path(temp)/'resume.json';cfg.write_text(json.dumps(config))
 captured=io.StringIO()
 with patch('sys.argv',['launch.py','--config',str(cfg),'--root',temp,'--gpus','0,1','--short']),contextlib.redirect_stdout(captured):
  launch.main()  # dry-run only: no GPU query or subprocess
 cmd=shlex.split(captured.getvalue())
 assert cmd[cmd.index('--resume')+1]==config['resume']
 target=int(cmd[cmd.index('--max-student-updates')+1])
 assert target==4

train=ast.parse((src/'train.py').read_text(encoding='utf-8'))
main=next(n for n in train.body if isinstance(n,ast.FunctionDef) and n.name=='main')
limit_assignment=next(n for n in main.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='limit' for t in n.targets))
loop=next(n for n in main.body if isinstance(n,ast.While))
env={'args':SimpleNamespace(benchmark=False,max_student_updates=target)}
exec(compile(ast.Module(body=[limit_assignment],type_ignores=[]),'audited_limit','exec'),env)
proof={}
for g in (0,1,3,4,40,160):
 env['gcount']=g
 entered=eval(compile(ast.Expression(loop.test),'audited_condition','eval'),env)
 proof[str(g)]={'enters_training_loop':entered,'new_G_updates_at_most':max(0,env['limit']-g)}
assert not proof['40']['enters_training_loop']
evidence={'source_sha256':hashes,'matches_previous_38_test_version':True,
          'new_Torch_tests_run':False,'new_standard_library_partition_cases':checks,
          'short_resume_dry_run':{'resume_retained':True,'absolute_limit':target,'starting_G_cases':proof},
          'gpu_jobs_started':False,'remote_access':False,'training_code_modified':False}
(out/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>修复版独立复审</title>
<style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:17px/1.75 system-ui,sans-serif;background:#f8fafc;color:#26354a}h1{font-size:30px}h2{font-size:22px;color:#225477;margin-top:30px}a{color:#1765a4}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d6dee9;padding:12px;text-align:left}.box{padding:16px;background:#fff2d8;border-left:5px solid #bc8125}code{background:#e8edf3;padding:2px 4px}</style>
<h1>修复版独立代码复审</h1><p>2026-10-22 · 三个独立审阅分工＋主审复核 · 当前 sdxl_research</p>
<p class="box"><strong>未发现新的 P1 或默认从头训练路径的确定性阻断问题。上轮三项修复均已生效。另确认 1 个 P2：配置含 resume 时使用 --short，可能零次更新却报告完成。</strong></p>
<h2>P2 · --short 与 resume 组合没有保证执行四次更新</h2>
<p><code>launch.py:57–60</code> 保留配置中的 resume，却将 max_student_updates 固定为 4。<code>train.py:230–232</code> 恢复已有 gcount；随后 <code>train.py:449–452</code> 把 4 当作绝对计数上限。</p>
<p>已用真实 launch.main() 的 dry-run 路径做标准库复现：含 G40 恢复路径的配置生成命令，同时保留 <code>--resume …g000040</code> 和 <code>--max-student-updates 4</code>。独立执行训练源码中的 limit 与 while 条件得到 <code>40 &lt; 4 == False</code>。因此在权重恢复及后续保存均成功的前提下，会跳过训练循环，继续写快照、完整状态，并将状态设为 complete。G1/G3 也只会再更新 3/1 次，而不是四次。</p>
<p>触发范围仅限有效、同协议的 resume 配置搭配 --short；默认无 resume 的短测仍执行四次。本轮是启动命令复现和源码路径核对，没有加载大权重模拟该错误。</p>
<p><strong>最小处理建议：</strong>--short 发现 resume 时明确拒绝，保持短测从头开始的简单语义。不需要引入额外恢复模式。本轮按审计范围没有修改它。</p>
<h2>上轮三项修复的独立核验</h2>
<table><tr><th>问题</th><th>复核结果</th></tr>
<tr><td>fake 阶段覆盖</td><td>fake 按 fcount 循环 1/2/3/4，学生按 gcount 循环 2/3/4，符合已下载官方源码相应分支。阶段 clean 输出送 fake；四步最终图送 GAN D；两者未串线。</td></tr>
<tr><td>轨迹复用与梯度</td><td>捕获中间 clean 不原地改变后续轨迹；tuple 的逐路 detach、拼接已正确。所有 rank 的计数、阶段与调用深度一致。无 GAN 路径不会使用 final=None。</td></tr>
<tr><td>CPU 卸载冲突</td><td>参数解析时拒绝双开；在加载模型前生效。新 fake_stage_policy 同时写入存档配置与恢复比对，旧协议不会静默接入。</td></tr>
<tr><td>评测分块续跑</td><td>独立执行当前分块函数，覆盖 6 种样本数 × 5 种旧 worker 数 × 5 种新 worker 数，共 150 组：分配无交集，完成后每个索引恰好一次。旧布局文件会明确拒绝。</td></tr></table>
<h2>验证边界与实验解释</h2><ul>
<li>当前 42 项源码/配置哈希与上一轮 38 项 CPU 测试通过的版本一致；本轮没有重跑 Torch 测试。新执行的是标准库分块检查及 --short 命令复现。</li>
<li>没有远程操作、GPU 任务、训练源码修改或对运行中实验的干预。</li>
<li>共享 fake 是阶段混合分布方案，不是每阶段独立条件 score；确认阶段策略不等于整个 SDXL 移植已等价复现视频论文。</li>
<li>joint34 仍用仅四步最终负样本训练的 D 指导第三步，是沿用的实验设计限制，未因本轮修复而变化。</li>
<li>换 worker 数续跑的前提是旧 worker 已退出；没有为同时写同一缓存增加调度系统。</li>
<li>真实预训练 SDXL 多卡数值、显存与训练效果仍需空闲资源上的短测，不能由静态检查代替。</li></ul>
<p><a href="evidence.json">本轮源码哈希与最小复现证据</a> · <a href="../audit_fixes_20261022/report.html">上一轮修复与 CPU 验证报告</a></p></html>'''
(out/'report.html').write_text(html,encoding='utf-8')
print(json.dumps({k:v for k,v in evidence.items() if k!='source_sha256'},ensure_ascii=False,indent=2))
