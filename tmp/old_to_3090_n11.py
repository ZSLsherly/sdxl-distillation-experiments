import sys,json,hashlib,shlex
from pathlib import PurePosixPath
sys.path.insert(0,'/data5/shenglin.zeng/tools/eval_transfer_20261025')
import paramiko
request=json.loads(sys.stdin.readline())
clients=[]
for ip,key in [('172.18.32.81','source'),('172.18.32.78','target')]:
 c=paramiko.SSHClient();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
 c.connect(ip,username=request[key]['user'],password=request[key].pop('password'),look_for_keys=False,allow_agent=False,timeout=20)
 c.get_transport().set_keepalive(20);clients.append(c)
source,target=clients
origin=PurePosixPath('/data3/shenglin.zeng/sdxl_turning_ablation_20261028/outputs/turning_batch64_g160_4gpu_teacher_shard_20261029')
root=PurePosixPath('/data1/shenglin.zeng/coco_missing_20261111')
groups={'normalized_u_g160':'20261031T233501+0800_g000160_periodic','raw_angle_g160':'20261103T044701+0800_g000160_periodic'}
with source.open_sftp() as sf,target.open_sftp() as tf:
 manifest={};made=set()
 for arm,stamp in groups.items():
  files=['config.json','status.json','data_manifest.json']
  files += ['weights/'+stamp+'/'+n for n in ['complete.json','shapes_rank0.json']+[f'student_rank{i}.pt' for i in range(4)]]
  for rel in files:
   src=origin/arm/rel;dst=root/'checkpoints'/arm/rel
   for parent in reversed(dst.parents):
    if str(parent) in made:continue
    try:tf.mkdir(str(parent))
    except IOError:tf.stat(str(parent))
    made.add(str(parent))
   _,output,error=source.exec_command('sha256sum -- '+shlex.quote(str(src)))
   digest=output.read().decode().split()[0];assert output.channel.recv_exit_status()==0
   size=sf.stat(str(src)).st_size;manifest[str(dst.relative_to(root))]=[size,digest]
   try:exists=tf.stat(str(dst)).st_size==size
   except IOError:exists=False
   if not exists:
    with tf.open(str(dst)+'.partial','wb') as out:
     out.set_pipelined(True)
     sf.getfo(str(src),out)
    assert tf.stat(str(dst)+'.partial').st_size==size
    tf.posix_rename(str(dst)+'.partial',str(dst))
   print('COPIED',arm,rel,size,flush=True)
 with tf.open(str(root/'transfer_manifest_old.json'),'w') as out:out.write(json.dumps(manifest))
print('OLD_TRANSFER_COMPLETE',len(manifest),flush=True)
