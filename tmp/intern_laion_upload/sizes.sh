set -eu
python3 - <<'PY'
import json,pathlib,urllib.request,urllib.parse,re
w=pathlib.Path('/tmp/laion-discovery-904.b9IQDB')
token=json.loads((w/'config/credentials').read_text())['token']
root=pathlib.Path('/data5/shenglin.zeng/sdxl_crossdistill/datasets/parallel_10k_seed20261018')
for name in ('manifest.json','prepared.pt'):
 req=urllib.request.Request('https://discovery.intern-ai.org.cn/api/datasets/v1/dataset/file/download_url',data=json.dumps(dict(did=904,version_id=723,path=name)).encode(),headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
 d=json.load(urllib.request.urlopen(req,timeout=20))
 payload=d['data']
 urls=[payload] if isinstance(payload,str) else [v for v in (payload or {}).values() if isinstance(v,str) and v.startswith('https://')]
 assert urls, (d.get('code'),list(payload or {}))
 with urllib.request.urlopen(urllib.request.Request(urls[0],headers={'Range':'bytes=0-0'}),timeout=25) as r:
  total=r.headers.get('Content-Range','').split('/')[-1] if r.status==206 else r.headers['Content-Length']
  print(name,'remote bytes',total,'local bytes',(root/name).stat().st_size)
  assert int(total)==(root/name).stat().st_size
PY
