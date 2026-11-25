set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import hashlib,json,os,subprocess
r=Path(os.environ['SDXL_ROOT']);n='A_pcm1_terminal_dm_g160_20261113';job=r/'outputs'/n;code=r/'tmp'/n/'code'
review=json.loads((job/'review.json').read_text());checked={}
for name,expected in review['source_sha256'].items():
 actual=hashlib.sha256((code/name).read_bytes()).hexdigest()
 checked[name]=actual
 assert actual==expected,(name,actual,expected)
print('DEPLOYED_HASHES_MATCH',len(checked))
for name in ['A_clean_global_g160_20261113',n]:
 p=r/'outputs'/name
 print('JOB',name,(p/'queue.log').read_text()[-600:])
 if (p/'model/status.json').exists():print((p/'model/status.json').read_text())
pid=int((job/'queue.pid').read_text());print('WAITER',pid,Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode())
print('CHILDREN',Path(f'/proc/{pid}/task/{pid}/children').read_text())
print('SOURCE_HASHES',json.dumps(checked))
PY
