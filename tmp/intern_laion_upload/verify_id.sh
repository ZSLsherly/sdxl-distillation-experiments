set -eu
python3 - <<'PY'
import json,urllib.request,urllib.error,pathlib
key=json.loads(pathlib.Path('/tmp/laion-discovery-904.b9IQDB/config/credentials').read_text())['token']
url='https://discovery.intern-ai.org.cn/api/datasets/v1/dataset/detail?did=904'
req=urllib.request.Request(url,headers={'Authorization':'Bearer '+key})
try:
 resp=json.load(urllib.request.urlopen(req,timeout=12)); print('response code',resp.get('code')); d=resp.get('data') or {}; print('fields',list(d));print({k:d.get(k) for k in ('did','id','name','dataset_name','dataset_id','warehouse_name','username','creator_name','last_version_id','version_id') if k in d})
except urllib.error.HTTPError as e: print('HTTP',e.code)
PY
