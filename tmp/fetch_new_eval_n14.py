"""Refresh small benchmark outputs through rclone; no checkpoint downloads."""
import os,json,subprocess,html
from pathlib import Path
from clip_all_remote_20261113 import auth
ROOT=Path('output/sdxl-crossdistill/new_models_eval_20261114')
RC=str(Path('tmp/rclone_runtime/rclone-v1.76.0-windows-amd64/rclone.exe').resolve())
nodes=json.loads((ROOT/'nodes.json').read_text());jobs=json.loads((ROOT/'jobs.json').read_text())
rows=[];summary={}
labels={'A_pcm10_low30':'A·PCM×10＋末步低噪声30%','A_clean_global':'A·中段clean x0＋全局加噪','A_terminal_only':'A·PCM×1＋仅末步学生DM（保留中段fake）'}
for host,node in nodes.items():
 key='N'+host.rsplit('.',1)[1];prefix='RCLONE_CONFIG_'+key+'_'
 env=dict(os.environ,RCLONE_CONFIG='NUL');env.update({prefix+'TYPE':'sftp',prefix+'HOST':host,prefix+'USER':auth[host][0],prefix+'PORT':'22'})
 env[prefix+'PASS']=subprocess.check_output([RC,'obscure','-'],input=auth[host][1]+'\n',text=True).strip()
 dest=ROOT/'snapshots'/host
 command=[RC,'copy',key.lower()+':'+node['root'],str(dest),'--log-level','ERROR','--retries','2']
 for pattern in ['/queue.log','/queue.pid','/传输记录.txt','/results/**/scores.json','/results/**/clip_results.json','/results/**/all_complete.json','/results/**/status_0.json','/results/**/worker.log']:command+=['--include',pattern]
 subprocess.run(command,env=env,check=True)
 for arm in node['arms']:
  r=dest/'results'/arm;final=r/'all_complete.json';status=r/'status_0.json'
  if final.exists():
   d=json.loads(final.read_text());assert d['fid_n']==10000 and d['clip_n']==1024
   state='完成';cells=[f"{d[k]:.6f}" for k in ('FID','Patch_FID','CLIP')]
  elif status.exists():
   d=json.loads(status.read_text());state=f"{d['done']}/10000";cells=['—']*3
  else:d={};state='准备/等待空闲卡';cells=['—']*3
  summary[arm]=dict(host=host,state=state,details=d)
  rows.append('<tr><td>'+html.escape(labels[arm])+'</td>'+''.join('<td>'+v+'</td>' for v in cells)+'<td>'+state+'</td></tr>')
  print(arm,state,d,flush=True)
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'report.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px Arial;max-width:1100px;margin:40px auto}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>新增三组G160：完整COCO评测</h1><p>FID与中心299×299 Patch FID：固定COCO2014 10K，越低越好。CLIP ViT-B/32：与历史结果相同的1024条提示词和种子，cosine×100，越高越好。</p><p>同一批10K生成图直接用于各指标，CLIP不重复生成。只保留原协议128张样图及特征。服务器间传输使用rclone，并核对SHA-256。</p><table><tr><th>模型</th><th>FID↓</th><th>Patch FID↓</th><th>CLIP↑</th><th>进度</th></tr>'+''.join(rows)+'</table><p>设备：172.18.32.81物理5号4090、172.18.32.78物理4号3090；禁用的4号4090不使用。遇到其他进程时退出自己的评测，等待后从已保存的特征块继续。</p>',encoding='utf-8')
