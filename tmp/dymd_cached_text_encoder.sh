python3 - <<'PY'
import pathlib,hashlib
for root in ['/data5/shenglin.zeng/models/Wan2.1-T2V-1.3B','/data5/shenglin.zeng']:
 p=pathlib.Path(root)/'models_t5_umt5-xxl-enc-bf16.pth'
 if p.exists():
  print('CACHED_FILE',str(p),p.stat().st_size,flush=True)
  h=hashlib.sha256()
  with p.open('rb') as f:
   while True:
    d=f.read(16*1024**2)
    if not d:break
    h.update(d)
  print('SHA256',h.hexdigest(),flush=True)
PY
