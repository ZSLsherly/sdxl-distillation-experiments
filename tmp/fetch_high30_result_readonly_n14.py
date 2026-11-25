"""Read completed neighboring evaluation only; no job or configuration changes."""
import os,json,hashlib,subprocess,shlex,html
from pathlib import Path
from clip_all_remote_20261113 import auth,connect
r=Path('output/sdxl-crossdistill/new_models_eval_20261114')
host='172.18.132.41';arm='A_clean_global_high30_v2'
remote='/data1/shenglin.zeng/high30_parallel_eval_20261114/results/'+arm
local=r/'snapshots'/host/'results'/arm
rc=str(Path('tmp/rclone_runtime/rclone-v1.76.0-windows-amd64/rclone.exe').resolve())
env=dict(os.environ,RCLONE_CONFIG='NUL',RCLONE_CONFIG_R41_TYPE='sftp',RCLONE_CONFIG_R41_HOST=host,RCLONE_CONFIG_R41_USER=auth[host][0],RCLONE_CONFIG_R41_PORT='22')
env['RCLONE_CONFIG_R41_PASS']=subprocess.check_output([rc,'obscure','-'],input=auth[host][1]+'\n',text=True).strip()
cmd=[rc,'copy','r41:'+remote,str(local),'--log-level','ERROR']
for pattern in ['/scores.json','/clip_results.json','/all_complete.json','/complete.json','/manifest.json','/status_*.json']:cmd+=['--include',pattern]
subprocess.run(cmd,env=env,check=True)
checks=[]
with connect(host) as c:
 for p in local.glob('*.json'):
  _,o,_=c.exec_command('sha256sum '+shlex.quote(remote+'/'+p.name));expected=o.read().decode().split()[0]
  actual=hashlib.sha256(p.read_bytes()).hexdigest();assert o.channel.recv_exit_status()==0 and expected==actual
  checks.append(dict(source_server=host,source=remote+'/'+p.name,destination=str(p.resolve()),sha256=actual,verified=True))
summary=json.loads((r/'summary.json').read_text(encoding='utf-8'))
new=json.loads((local/'all_complete.json').read_text());assert new['n']==10000 and new['clip_n']==1024
summary[arm]=dict(host=host,state='完成',details=new)
selection=json.loads((r/'selection.json').read_text(encoding='utf-8'))
lines=[];rows=[]
labels={'A_clean_global':'A·中段clean x0＋全局均匀加噪','A_terminal_only':'A·仅末步学生DM（保留中段fake）','A_pcm10_low30':'A·PCM×10＋末步低噪声30%','A_clean_global_high30_v2':'clean全局＋高噪声30% v2'}
for a,entry in summary.items():
 p=r/'snapshots'/entry['host']/'results'/a
 d=json.loads((p/'all_complete.json').read_text());q=json.loads((p/'clip_results.json').read_text())
 assert d.get('fid_n',d.get('n'))==10000 and d['clip_n']==1024
 assert q['cases']==selection['cases'] and q['revision']==selection['revision']
 values=q['per_case'][a]['alignment'];assert len(values)==1024 and abs(sum(values)/1024-d['CLIP'])<1e-4
 cols=[labels[a],f"{d['FID']:.6f}",f"{d['Patch_FID']:.6f}",f"{d['CLIP']:.6f}",entry['host'],'10000 / 1024']
 rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cols)+'</tr>');lines.append(' | '.join(cols))
(r/'summary_four_complete.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'四组完成与校验记录.txt').write_text('四组完整COCO评测已完成。FID/PatchFID各10000，CLIP固定1024；四组CLIP提示词/种子/revision一致。\n'+'\n'.join(lines)+'\n邻居节点小型结果通过rclone取回，SHA256核对如下：\n'+json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'report_four_complete.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px Arial;max-width:1250px;margin:40px auto}th,td{padding:12px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style><h1>四组新实验：完整COCO评测已完成</h1><p>FID与中心299×299 Patch FID：固定COCO2014 10K，越低越好。CLIP ViT-B/32：历史固定1024提示词/种子，cosine×100，越高越好。四组样本数量及CLIP逐样本记录已核对。</p><table><tr><th>实验</th><th>FID↓</th><th>Patch FID↓</th><th>CLIP↑</th><th>评测服务器</th><th>FID / CLIP 样本数</th></tr>'+''.join(rows)+'</table><p>高噪声v2相对clean全局基线：FID和Patch FID均升高，CLIP均值略升。不能据此认定高噪声先验带来整体质量改善。</p>',encoding='utf-8')
print('VERIFIED_FOUR_COMPLETE')
for line in lines:print(line)
