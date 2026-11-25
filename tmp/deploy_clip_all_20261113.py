from pathlib import Path
import json,sys
from clip_all_remote_20261113 import connect
root=Path('output/sdxl-crossdistill/clip_all_20261113');root.mkdir(parents=True,exist_ok=True)
base=Path('output/sdxl-crossdistill')
sources=[('four',base/'semantic_four_1024_20261103/results.json'),('pair',base/'semantic_pair_1024_20261105/results.json'),('old_uniform',base/'semantic_old_uniform_20261104/results.json'),('sdxl',base/'all_experiments_20261104/sdxl_results.json'),('dmd2',base/'all_experiments_20261104/dmd2_results.json')]
main=json.loads(sources[0][1].read_text());cases=main['cases'];revision=main['revision']
existing=dict(n=1024,revision=revision,cases=cases,metrics={},per_case={})
for _,p in sources:
 d=json.loads(p.read_text());assert d['revision']==revision and d['cases']==cases,(p,'protocol mismatch')
 for name,metrics in d['metrics'].items():
  if name in existing['metrics']:assert existing['metrics'][name]==metrics
  existing['metrics'][name]=metrics
  if name in d.get('per_case',{}):existing['per_case'][name]=d['per_case'][name]
assert len(existing['metrics'])==9,existing['metrics'].keys()
(root/'existing_results.json').write_text(json.dumps(existing,ensure_ascii=False,indent=2),encoding='utf-8')
(root/'selection.json').write_text(json.dumps(dict(revision=revision,cases=cases),ensure_ascii=False),encoding='utf-8')
c=connect('172.18.32.81');remote='/data3/shenglin.zeng/clip_missing_20261113'
_,o,_=c.exec_command('mkdir -p '+remote+'/code '+remote+'/existing');assert o.channel.recv_exit_status()==0
files=[('tmp/clip_all_worker_20261113.py','code/worker.py'),('tmp/clip_all_queue_20261113.py','code/run_queue.py'),('sdxl_research/evaluation_core.py','code/evaluation_core.py'),('sdxl_research/objectives.py','code/objectives.py'),(root/'selection.json','selection.json'),(root/'existing_results.json','existing_results.json'),(sources[2][1],'existing/old_uniform.json')]
with c.open_sftp() as s:
 try:s.remove(remote+'/code/queue.py')
 except FileNotFoundError:pass
 for local,dest in files:s.put(str(local),remote+'/'+dest)
 s.put('tmp/clip_all_report_20261113.py',remote+'/code/report.py')
print('DEPLOYED',len(files),'existing',list(existing['metrics']))
c.close()
