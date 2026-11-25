"""One-shot, score-gated cleanup of three named A checkpoints on this node."""
import fcntl,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path

archive=Path('/data/shenglin.zeng/a50k_20261123')
report=archive/'eval_A_milestones_20261125'
training=Path('/data5/shenglin.zeng/a50k_20261124')
mode=sys.argv[1]
assert mode in ('archive','training')
base=(archive/'checkpoints/singlepass' if mode=='archive' else training/'outputs/A50K_singlepass/training_state')
assert base.resolve()==base
location=report if mode=='archive' else training/'logs'
lock=(location/'prune_scored_A.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if mode=='training':
    auth=json.loads(sys.stdin.readline())
    exe='/data5/shenglin.zeng/checkpoints/a800_archive_20261114/.tools/rclone'
    env=dict(os.environ,RCLONE_CONFIG='/dev/null',RCLONE_CONFIG_ARCH_TYPE='sftp',RCLONE_CONFIG_ARCH_HOST='172.18.32.81',RCLONE_CONFIG_ARCH_USER=auth['u'],RCLONE_CONFIG_ARCH_PASS=subprocess.check_output([exe,'obscure','-'],input=auth.pop('p')+'\n',universal_newlines=True).strip())
if os.fork():sys.exit(0)
os.setsid()
log=(location/'A基线中间存档清理记录.txt').open('a',1)
os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
with open('/dev/null') as fd:os.dup2(fd.fileno(),0)
def note(*args):print(time.strftime('%F %T %z'),*args,flush=True)
note('Waiting for four COCO10K scores and CLIP1024 verification')
while True:
    if mode=='archive':
        if (report/'summary.json').is_file():scores=json.loads((report/'summary.json').read_text());break
    else:
        q=subprocess.run([exe,'cat','arch:'+str(report/'archive_pruned.json'),'--retries','1'],env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        if q.returncode==0:scores=json.loads(q.stdout)['scores'];break
    time.sleep(60)
steps={20000:313,30000:470,40000:627,50000:784}
files={f'{role}_rank{i}.pt' for role in ('student','fake','optimizer_og','optimizer_of','rng') for i in range(4)}|{'complete.json'}
dirs={}
for n,g in steps.items():
    arm=f'A{n}';s=scores[arm]
    assert s['arm']==arm and s['fid_n']==10000 and s['clip_n']==1024 and all(math.isfinite(s[k]) for k in ('FID','Patch_FID','CLIP'))
    p=base/f'samples{n:05d}_g{g:06d}';dirs[n]=p
    assert p.resolve()==p and p.parent==base and p.is_dir()
    assert {f.name for f in p.iterdir()}==files and all(f.is_file() and not f.is_symlink() and f.stat().st_size for f in p.iterdir())
    m=json.loads((p/'complete.json').read_text());assert m['world_size']==4 and m['config']['run']=='A50K_singlepass'
    assert m['progress']==dict(gcount=g,fcount=g*2,dcount=0,student_samples=n)
    if mode=='archive':
        assert json.loads((report/'results'/arm/'all_complete.json').read_text())==s
        assert len(list((report/'results'/arm/'images'/arm).glob('*.png')))==128
assert hashlib.sha256((dirs[50000]/'complete.json').read_bytes()).hexdigest()=='55615c3fff7fb9ac526878327b2b11d9932c1e58a9ba074c67997fc9bf637288'
deleted=[]
for n in [20000,30000,40000]:
    p=dirs[n];size=sum(f.stat().st_size for f in p.iterdir())
    for filename in sorted(files): (p/filename).unlink()
    p.rmdir();deleted.append(str(p));note('DELETED',p,size)
note('RETAINED',dirs[50000],'SCORES',scores)
receipt=location/(mode+'_pruned.json')
temp=receipt.with_suffix('.tmp');temp.write_text(json.dumps(dict(scores=scores,deleted=deleted,retained=str(dirs[50000])),indent=2));temp.replace(receipt)
