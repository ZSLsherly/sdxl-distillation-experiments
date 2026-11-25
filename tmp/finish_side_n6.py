from backup_a800_nov5 import cloud,lab,run,q,ROOT
from abc_direct_grant import store
from pathlib import Path
import json,datetime
LAB='/data5/shenglin.zeng/checkpoints/a800_archive_20261106'
STAGE='/root/sdxl_research/tmp/side_archive_n6'
local=Path('output/sdxl-crossdistill/a800_archive_20261106');local.mkdir(exist_ok=True)
with cloud() as s,lab() as d:
 complete=json.loads(run(d,'cat '+q(LAB+'/COMPLETE.json')))
 m=json.loads(run(d,'cat '+q(LAB+'/manifest.json')))
 assert complete['files']==len(m['records'])
 weights=[k for k in m['large'] if k.endswith('/student_rank0.pt')]
 data=['datasets/parallel_20k_seed20261018/prepared.pt','datasets/20k_sources/vae_latents_002.pt','datasets/20k_sources/vae_latents_003.pt']
 assert len(weights)==2
 targets=weights+data
 for k in targets:
  n,h=m['records'][k]
  assert run(d,'realpath '+q(LAB+'/'+k))==LAB+'/'+k
  assert int(run(d,'stat -c %s '+q(LAB+'/'+k)))==n
  assert run(d,'sha256sum '+q(LAB+'/'+k)).split()[0]==h
  print('INDEPENDENT_LAB_HASH_OK',k,flush=True)
 # Place all verified small artifacts in normal directories for future use.
 unpack="""from pathlib import Path
import tarfile,json,hashlib
r=Path(LAB);m=json.loads((r/'manifest.json').read_text())
for i,rels in enumerate(m['groups']):
 with tarfile.open(r/'evidence'/f'part_{i:03d}.tar',mode='r|') as t:
  seen=[]
  for a in t:
   assert a.isfile() and a.name in rels
   p=r/a.name;p.resolve().relative_to(r)
   b=t.extractfile(a).read();assert len(b)==m['records'][a.name][0] and hashlib.sha256(b).hexdigest()==m['records'][a.name][1]
   p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert p.read_bytes()==b
   else:p.write_bytes(b)
   seen.append(a.name)
  assert seen==rels
print('ALL_SMALL_ARTIFACTS_RESTORED')
""".replace('LAB',repr(LAB))
 store(d,LAB+'/.pull6/unpack.py',unpack,0o700)
 print(run(d,'python3 '+LAB+'/.pull6/unpack.py'),flush=True)
 # Remove only the five explicit, now independently verified large source files.
 payload={'root':ROOT,'lab':LAB,'targets':{k:m['records'][k] for k in targets},'group_count':len(m['groups'])}
 cleaner="""from pathlib import Path
import hashlib,json,os,subprocess
c=PAYLOAD;r=Path(c['root'])
live=subprocess.check_output(['ps','-eo','pid,args'],text=True)
for token in ['parallel_20k_seed20261018','20k_sources','A_pcm10_terminal_patchgan','A_mid_clean_x0']:
 assert not any(token in line for line in live.splitlines() if ('python' in line or 'torchrun' in line)),token
removed=[]
for rel,(size,h) in c['targets'].items():
 p=r/rel;assert p.resolve()==p and p.is_file() and p.stat().st_size==size
 sha=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):sha.update(b)
 assert sha.hexdigest()==h,rel
 receipt=p.with_name(p.name+'.LAB_BACKUP.txt')
 receipt.write_text('Verified laboratory backup: '+c['lab']+'/'+rel+'\\nSHA256: '+h+'\\nBytes: '+str(size)+'\\n',encoding='utf-8')
 p.unlink();removed.append(rel)
stage=r/'tmp/side_archive_n6/groups'
for i in range(c['group_count']):
 p=stage/f'part_{i:03d}.tar'
 assert p.resolve()==p and p.is_file()
 p.unlink()
print(json.dumps({'removed':removed,'freed_bytes':sum(v[0] for v in c['targets'].values())}))
""".replace('PAYLOAD',repr(payload))
 store(s,STAGE+'/verified_cleanup.py',cleaner,0o700)
 result=json.loads(run(s,'/root/sdxl_research/env/bin/python '+STAGE+'/verified_cleanup.py'))
 run(s,'/root/sdxl_research/env/bin/python '+STAGE+'/revoke.py')
 run(d,'unlink '+LAB+'/.pull6/key')
 free=run(s,'df -h /root/autodl-tmp; ps -eo pid,args | grep -E "train.py|run_A10_mix1" | grep -v grep || true')
 receipt=dict(timestamp=datetime.datetime.now().isoformat(),destination=LAB,complete=complete,cleanup=result,files=payload['targets'],state=free)
 (local/'receipt.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
 txt='2026-11-06 A800备份清理：两个已完成实验A_pcm10_terminal_patchgan和A_mid_clean_x0（含权重、日志、评测和可发现的代码快照）及20K数据已直传 '+LAB+'；完整清单逐文件SHA-256校验，五个大文件另做独立实验室全哈希核对后删除A800源件，共释放'+str(round(result['freed_bytes']/2**30,2))+'GiB。当前A_pcm10_mix1_g160的10K训练依赖、原始模型、日志、图像和评分保留；临时直传授权已撤销。\n'
 (local/'备份清理记录.txt').write_text(txt+'\n'+free,encoding='utf-8')
 with Path('output/sdxl-crossdistill/实验记录.txt').open('a',encoding='utf-8') as f:f.write('\n'+txt)
 store(d,LAB+'/备份清理记录.txt',txt,0o600)
 print(json.dumps(result),flush=True);print(free,flush=True)
