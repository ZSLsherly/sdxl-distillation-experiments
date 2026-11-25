"""Fetch this run's status and publish its local experiment report."""
from pathlib import Path
import sys,re,json,time,html,os
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];os.chdir(ROOT)
sys.path.insert(0,str(ROOT/'.image_runtime_pkgs'))
import paramiko
OUT=ROOT/'output/sdxl-crossdistill/joint34_uniform_a100_20261025';OUT.mkdir(parents=True,exist_ok=True)
REMOTE='/data5/shenglin.zeng/sdxl_crossdistill/outputs/joint34_uniform_10k_2gpu_20261025'
def sync():
 raw=Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
 u,p=next((u,p.strip()) for ip,u,p in re.findall(r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)',raw) if ip=='172.18.33.60')
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy());c.connect('172.18.33.60',username=u,password=p,timeout=20,look_for_keys=False,allow_agent=False)
 state={'checked_unix':time.time(),'phases':{}}
 with c.open_sftp() as s:
  def read(name,limit=None):
   try:
    with s.open(REMOTE+'/'+name) as f:
     size=f.stat().st_size;partial=bool(limit and size>limit)
     if partial:f.seek(size-limit)
     text=f.read().decode('utf-8')
     return text.split('\n',1)[1] if partial else text
   except FileNotFoundError:return None
  for name in ['queue_status.json','design.json','phase1_config.json','train_config.json','workflow_complete.json','workflow_stopped.json','combined_scores.json','coco10k/scores.json','coco10k/complete.json']:
   text=read(name)
   if text is not None:
    obj=json.loads(text);state[name]=obj;(OUT/name.replace('/','_')).write_text(text,encoding='utf-8')
  def jsonlines(name):
   rows=[]
   for line in (read(name,128*1024) or '').splitlines():
    try:rows.append(json.loads(line))
    except json.JSONDecodeError:pass
   return rows
  state['events']=jsonlines('workflow.jsonl')[-8:]
  resource=jsonlines('resource_monitor.jsonl');state['last_resource']=resource[-1] if resource else None
  for phase in ('phase1','train'):
   d={}
   for name in ('status.json','config.json','source_sha256.json'):
    text=read(phase+'/'+name)
    if text is not None:d[name]=json.loads(text)
   rows=jsonlines(phase+'/rank0.jsonl');gs=[r for r in rows if r['phase']=='student_done']
   d['latest_student']=gs[-1] if gs else None;d['latest_update']=rows[-1] if rows else None
   if d.get('config.json') or rows:state['phases'][phase]=d
  for rank in (0,1):
   text=read(f'coco10k/status_{rank}.json')
   if text is not None:state.setdefault('evaluation_workers',[]).append(json.loads(text))
 c.close();(OUT/'status_snapshot.json').write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf-8')
 return state
def report(s):
 stamp=datetime.fromtimestamp(s['checked_unix'],timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
 phases=s['phases'];active=phases.get('train',phases.get('phase1',{}));st=active.get('status.json',{});latest=active.get('latest_student') or {}
 g=max(st.get('student_updates',0),latest.get('g',0));label=f'训练进行中：G{g}/160'
 queue=s.get('queue_status.json',{})
 gpu_text=','.join(map(str,queue.get('gpus',[]))) or '等待空闲双卡'
 if queue.get('state')=='waiting':label='排队等待空闲双卡；当前空闲卡：'+str(queue.get('free_gpus',[]))
 if 'workflow_stopped.json' in s:label='流程停止：'+s['workflow_stopped.json']['reason']
 elif 'workflow_complete.json' in s:label='G160 与完整 COCO10K 评测均已完成'
 elif s.get('evaluation_workers'):label=f"G160 已完成；COCO10K 已提交 {sum(x['done'] for x in s['evaluation_workers']):,}/10,000 张"
 elif any(e.get('phase')=='evaluation' for e in s['events']):label='G160 已完成；正在准备或执行 COCO10K 评测'
 scores=json.loads((ROOT/'output/sdxl-crossdistill/coco_uniform_g160_20261025/combined_scores.json').read_text())
 scores.update(s.get('coco10k/scores.json',{}))
 names={'joint4_uniform':'均匀四点：第4步 GAN+DMD（参考组）','joint34_uniform':'均匀四点：第3、4步 GAN+DMD（本组）','dmd_only':'旧四点：纯 DMD','gan4':'旧四点：第4步纯 GAN','joint4':'旧四点：第4步 GAN+DMD','joint34':'旧四点：第3、4步 GAN+DMD','dmd2':'官方 DMD2，4步','sdxl':'原始 SDXL，50步'}
 rows=''
 for key,name in names.items():
  v=scores.get(key);cells=f"<td>{v['FID']:.4f}</td><td>{v['Patch_FID']:.4f}</td>" if v else '<td>待完整评测</td><td>待完整评测</td>'
  rows+=f'<tr><td>{name}</td>{cells}</tr>'
 delta=''
 if 'joint34_uniform' in scores:
  a,b=scores['joint34_uniform'],scores['joint4_uniform'];delta=f"本组相对均匀四点末步联合组：FID 变化 {a['FID']-b['FID']:+.4f}，Patch FID 变化 {a['Patch_FID']-b['Patch_FID']:+.4f}（负值更好）。"
  (OUT/'combined_scores.json').write_text(json.dumps(scores,indent=2))
 progress=''
 if latest:progress=f"最近完成 G{latest['g']}，监督第 {latest['stage']} 步，GAN 生效：{latest['gan']}；完整周期 {latest['cycle_seconds']:.1f} 秒，学生梯度范数 {latest['grad_norm']:.4f}。"
 refresh='' if 'workflow_complete.json' in s or 'workflow_stopped.json' in s else '<meta http-equiv="refresh" content="60">'
 page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">{refresh}<title>均匀四点 joint34 G160 对照</title><style>body{{max-width:1120px;margin:35px auto;padding:0 24px;background:#f5f7fa;color:#183047;font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif}}section{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:22px;margin:22px 0}}h1{{font-size:28px}}h2{{font-size:21px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #e2e8f0;padding:9px}}.note{{color:#596b80}}a{{color:#1763ae}}</style>
<h1>均匀四点：第3、4步 GAN+DMD 对照</h1><p>最后同步：北京时间 {stamp}</p><section><h2>{html.escape(label)}</h2><p>{progress}</p><p>服务器 172.18.33.60，GPU {gpu_text}。等待至少两张卡连续空闲60秒且主机可用内存≥120 GiB；最长等待24小时，期间不占用GPU。启动前再次检查空闲，检测到其他进程进入时仅停止自己的任务。核心训练源码与参考组相同，未改动正在运行的其他任务。</p></section>
<section><h2>对照设计</h2><p><b>唯一计划改变的训练目标设置：GAN 生效阶段 [4] → [3,4]。</b>从原始 SDXL 独立初始化学生、fake 与判别器，不从参考组 G160 续训。</p>
<table><tr><th>设置</th><th>两组共同设置</th></tr><tr><td>采样点 / PCM</td><td>999 / 749 / 499 / 249；PCM 边界 749、权重 10</td></tr><tr><td>DMD</td><td>第2、3、4步；再加噪时间上限 919；fake 按 1/2/3/4 轮换</td></tr><tr><td>数据 / 随机种子</td><td>完全相同的 10K LAION 衍生数据；seed=20261017</td></tr><tr><td>batch</td><td>全局 64、双卡；G1–2 每卡4/累积8，G3–160 每卡16/累积2；rollout batch 同步4→16</td></tr><tr><td>学习率</td><td>学生/fake 1e-6；D 1e-4</td></tr><tr><td>GAN</td><td>系数 0.00527422968；hinge；DINO+RGB 判别器；局部 latent 解码；D-only 翻转</td></tr><tr><td>CFG / 更新预算</td><td>老师8、学生1；G160，fake800，D800；fake:G=5:1</td></tr><tr><td>保存与过程评估</td><td>G40/80/120/160 保存学生权重；最终完整可续训状态；每128个学生样本进行固定小样本监控</td></tr></table>
<p>本组 GAN 学生更新为 106 次，参考组为 53 次；其余目标更新次数不变。第3步 GAN 监督的是第三步的 clean 预测，并非 noisy handoff latent；判别器仍用最终四步样本训练，保持参考设置。中间 G2 完整状态用于复刻批量切换，在最终续训状态和 COCO10K 均成功后删除。</p>
<p class="note">本组恢复为参考组的原A100服务器、原环境和双卡布局，训练算法、CPU offload与批量切换均保持一致。此前4090尝试在首个学生更新完成前停止，发现共享用户内存软限100 GiB、硬限112 GiB；四卡备选也因其他任务已用满软限而取消预检，未启动四卡训练。未修改服务器内存限额。</p></section>
<section><h2>这组能回答什么</h2><p>它检验“在已有末步 GAN 的基础上，再加入第3步 GAN 是否有收益”。要判断 GAN 本身是否优于无 GAN，需要另做同配置的均匀四点纯 DMD 对照。本次未额外启动该组。单种子差异也不等同于稳定的统计显著收益。</p></section>
<section><h2>完整 COCO10K 对照</h2><p>G160 后自动评测固定 COCO2014 10K，沿用相同提示词、seed、1024分辨率、uint8转换、Inception及预处理；真实图特征复用。只将完整10K分数并入下表。</p><table><tr><th>方案</th><th>FID ↓</th><th>Patch FID ↓</th></tr>{rows}</table><p>{delta}</p><p>Patch FID 为中心裁剪分布距离，不能单独证明纹理真实且正确。</p></section>
<section><h2>可复核文件</h2><p><a href="status_snapshot.json">状态与实际配置</a> · <a href="design.json">设计记录</a> · <a href="../coco_uniform_g160_20261025/report.html">参考组完整结果</a></p><p>远端目录：{REMOTE}</p></section></html>'''
 (OUT/'report.html').write_text(page,encoding='utf-8');print(label,flush=True)
 return 'workflow_complete.json' in s or 'workflow_stopped.json' in s
deadline=time.monotonic()+50*3600
while True:
 try:done=report(sync())
 except Exception as exc:
  print('SYNC_FAILED',type(exc).__name__,str(exc),flush=True);done=False
  if '--watch' not in sys.argv:raise
 if '--watch' not in sys.argv or done or time.monotonic()>deadline:break
 time.sleep(300)
