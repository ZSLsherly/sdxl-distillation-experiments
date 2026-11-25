import os,sys,json,hashlib,subprocess,shutil,shlex
from pathlib import Path
from clip_all_remote_20261113 import connect,auth
local=Path('output/sdxl-crossdistill/new_models_eval_20261114');code=local/'code';code.mkdir(exist_ok=True)
for src,name in [('tmp/new_eval_worker_n14.py','worker.py'),('tmp/new_eval_queue_n14.py','run_queue.py'),('tmp/new_eval_stage_n14.py','stage.py'),('tmp/new_eval_prepare_node_n14.py','prepare_node.py'),('sdxl_research/evaluation_core.py','evaluation_core.py'),('sdxl_research/objectives.py','objectives.py')]:shutil.copy2(src,code/name)
shutil.copy2('output/sdxl-crossdistill/clip_all_20261113/selection.json',local/'selection.json')
nodes={
'172.18.32.81':dict(host='172.18.32.81',root='/data3/shenglin.zeng/new_models_eval_20261114',python='/data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python',gpu='GPU-5eceb5c0-44bb-9b56-d40e-2846ce7ff00f',arms=['A_clean_global','A_terminal_only'],links={'models/sdxl-base-1.0':'/data3/shenglin.zeng/clip_missing_20261113/models/sdxl-base-1.0','models/clip':'/data3/shenglin.zeng/clip_missing_20261113/models/clip'}),
'172.18.32.78':dict(host='172.18.32.78',root='/data1/shenglin.zeng/new_models_eval_20261114',python='/data1/shenglin.zeng/teacher_turning_20261026/env/bin/python',gpu='GPU-b725e1ea-ce0d-1662-ec47-af52200242eb',arms=['A_pcm10_low30'],links={'models/sdxl-base-1.0':'/data1/shenglin.zeng/coco_missing_20261111/models/sdxl-base-1.0','models/metrics/clean_fid':'/data1/shenglin.zeng/coco_missing_20261111/models/metrics/clean_fid','reference/features/real':'/data1/shenglin.zeng/coco_missing_20261111/reference/features/real'})}
(local/'nodes.json').write_text(json.dumps(nodes,indent=2),encoding='utf-8')
RC=str(Path('tmp/rclone_runtime/rclone-v1.76.0-windows-amd64/rclone.exe').resolve());host='172.18.33.60'
env=dict(os.environ,RCLONE_CONFIG='NUL',RCLONE_CONFIG_LAB_TYPE='sftp',RCLONE_CONFIG_LAB_HOST=host,RCLONE_CONFIG_LAB_PORT='22',RCLONE_CONFIG_LAB_USER=auth[host][0])
env['RCLONE_CONFIG_LAB_PASS']=subprocess.check_output([RC,'obscure','-'],input=auth[host][1]+'\n',text=True).strip()
remote='/data5/shenglin.zeng/tools/new_models_eval_20261114'
subprocess.run([RC,'copy',str(local),'lab:'+remote,'--exclude','__pycache__/**','--log-level','ERROR','--retries','3'],env=env,check=True)
c=connect(host)
checks=[]
for p in local.rglob('*'):
 if not p.is_file() or '__pycache__' in p.parts:continue
 dst=remote+'/'+p.relative_to(local).as_posix();_,o,_=c.exec_command('sha256sum '+shlex.quote(dst));got=o.read().decode().split()[0]
 digest=hashlib.sha256(p.read_bytes()).hexdigest();assert o.channel.recv_exit_status()==0 and digest==got
 checks.append(dict(local=str(p.resolve()),destination_server=host,destination=dst,sha256=digest))
(local/'code_deploy_sha256.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
print('DEPLOY_SHA256_VERIFIED',len(checks),flush=True)
w,o,_=c.exec_command('/data5/shenglin.zeng/envs/python-runtime/bin/python3.12 -u '+shlex.quote(remote+'/code/stage.py'));o.channel.set_combine_stderr(True)
w.write(json.dumps({h:auth[h] for h in nodes})+'\n');w.flush();w.channel.shutdown_write()
for l in o:print(l.rstrip(),flush=True)
sys.exit(o.channel.recv_exit_status())
