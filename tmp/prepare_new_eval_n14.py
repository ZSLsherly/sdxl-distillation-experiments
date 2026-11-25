"""Collect small source metadata with rclone; never download student weights locally."""
import os,sys,json,hashlib,subprocess,shlex
from pathlib import Path
from clip_all_remote_20261113 import connect,auth,paramiko
ROOT=Path('output/sdxl-crossdistill/new_models_eval_20261114');ROOT.mkdir(parents=True,exist_ok=True)
RC=str(Path('tmp/rclone_runtime/rclone-v1.76.0-windows-amd64/rclone.exe').resolve())
password=sys.stdin.readline().strip()
env=dict(os.environ,RCLONE_CONFIG='NUL',RCLONE_CONFIG_A8_TYPE='sftp',RCLONE_CONFIG_A8_HOST='connect.nma1.seetacloud.com',RCLONE_CONFIG_A8_PORT='14892',RCLONE_CONFIG_A8_USER='root')
env['RCLONE_CONFIG_A8_PASS']=subprocess.check_output([RC,'obscure','-'],input=password+'\n',text=True).strip()
src=paramiko.SSHClient();src.set_missing_host_key_policy(paramiko.AutoAddPolicy());src.connect('connect.nma1.seetacloud.com',port=14892,username='root',password=password,timeout=20,look_for_keys=False,allow_agent=False)
archive=Path('output/sdxl-crossdistill/a800_archive_20261114_side');old=json.loads((archive/'manifest.json').read_text())
groups={k:dict(weight=v['weight'],sha256=old['records'][v['weight']]['sha256']) for k,v in old['groups'].items()}
receipt=json.loads((archive/'A_terminal_only_receipt.json').read_text())
groups[receipt['group']]=dict(weight=receipt['source'].removeprefix('/root/sdxl_research/'),sha256=receipt['sha256'])
names={'A_pcm10_terminal_low30_g160_20261113':('A_pcm10_low30','172.18.32.78'),'A_clean_global_g160_20261113':('A_clean_global','172.18.32.81'),'A_pcm1_terminal_dm_g160_20261113':('A_terminal_only','172.18.32.81')}
jobs={};checks=[]
for group,v in groups.items():
 arm,host=names[group];dest=ROOT/'metadata'/arm;dest.mkdir(parents=True,exist_ok=True)
 checkpoint=str(Path(v['weight']).parent).replace('\\','/')
 files={'config.json':f'outputs/{group}/model/config.json','status.json':f'outputs/{group}/model/status.json','complete.json':checkpoint+'/complete.json','shapes_rank0.json':checkpoint+'/shapes_rank0.json','cpu_validation.json':checkpoint+'/cpu_validation.json'}
 for name,relative in files.items():
  remote='/root/sdxl_research/'+relative
  subprocess.run([RC,'copyto','a8:'+remote,str(dest/name),'--retries','3','--log-level','ERROR'],env=env,check=True)
  _,o,_=src.exec_command('sha256sum '+shlex.quote(remote));expected=o.read().decode().split()[0];assert o.channel.recv_exit_status()==0
  got=hashlib.sha256((dest/name).read_bytes()).hexdigest();assert got==expected
  checks.append(dict(source_server='connect.nma1.seetacloud.com:14892',source=remote,destination=str((dest/name).resolve()),sha256=got,verified=True))
 cfg=json.loads((dest/'config.json').read_text());status=json.loads((dest/'status.json').read_text());complete=json.loads((dest/'complete.json').read_text());validation=json.loads((dest/'cpu_validation.json').read_text())
 assert status['status']=='complete' and status['student_updates']==160 and status['fake_updates']==320
 assert complete['student_updates']==160 and complete['world_size']==1
 assert validation['strict_load'] and validation['finite']
 jobs[arm]=dict(group=group,host=host,source='/data5/shenglin.zeng/checkpoints/a800_archive_20261114/'+v['weight'],sha256=v['sha256'],config=cfg,world_size=1,student_updates=160)
 print('VERIFIED_METADATA',arm,cfg['schedule'],flush=True)
(ROOT/'jobs.json').write_text(json.dumps(jobs,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'metadata_checksums.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'评测与传输记录.txt').write_text('新增三组G160评测；来源：已完成训练的config/status/complete/cpu_validation及实验室权重备份。\n采用rclone传输，小型元数据逐文件SHA256核对一致；模型权重不落盘本机。\nCOCO2014固定10K：FID、中心299x299 Patch FID；同一固定COCO1024：CLIP ViT-B/32、Top1/Top5。\n物理GPU：172.18.32.81的5号4090；172.18.32.78的4号3090。排除禁用4号4090。\n高噪声v2尚在训练，不使用未完成权重。\n'+json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
src.close()
