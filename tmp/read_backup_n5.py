from pathlib import Path
import os,re,json,sys,time
stage=Path(__file__).parent;root=Path('/root/autodl-tmp/sdxl_runs')
left=int((stage/'expiry').read_text())-int(time.time())
if left<=0:raise SystemExit(77)
cmd=os.environ.get('SSH_ORIGINAL_COMMAND','')
m=json.loads((stage/'manifest.json').read_text())
if cmd=='manifest':args=['cat',str(stage/'manifest.json')]
elif v:=re.fullmatch(r'g (\d+)',cmd):
 i=int(v[1]);assert i<len(m['groups'])
 args=['tar','-C',str(root),'-cf','-','--no-recursion','--null','-T',str(stage/f'list{i}')]
elif v:=re.fullmatch(r'b (\d+) (\d+) (\d+)',cmd):
 i,off,n=map(int,v.groups());rel=m['large'][i];assert 0<n<=64*2**20 and off+n<=m['records'][rel][0]
 args=['dd','if='+str(root/rel),'bs=1M','iflag=skip_bytes,count_bytes,fullblock','skip='+str(off),'count='+str(n),'status=none']
else:raise SystemExit(77)
os.execvp('timeout',['timeout',str(min(1800,left)),*args])

