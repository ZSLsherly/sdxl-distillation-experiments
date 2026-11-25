import ssl,socket,sys,os,struct,hashlib,json,time
from pathlib import Path
# Single-use authenticated transfer. Receiver supplies the entire filename allowlist.
send=sys.argv[1]=='send'
job=Path.cwd()
if send:
 base=Path('/data5/shenglin.zeng/sdxl_crossdistill')
 root=base/'outputs/coco10k_g160_20261020'
 ids=json.loads((root/'manifest.json').read_text())['visual_indices']
 ctx=ssl.create_default_context(cafile='peer.crt')
 ctx.load_cert_chain('local.crt','local.key')
 raw=socket.create_connection(('172.18.132.41',int(sys.argv[2])),timeout=30)
 sock=ctx.wrap_socket(raw,server_hostname='sem-recv')
else:
 root=job.parent
 ids=json.loads((root/'ref.json').read_text())['visual_indices']
 ctx=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
 ctx.load_cert_chain('local.crt','local.key');ctx.load_verify_locations(cafile='peer.crt')
 ctx.verify_mode=ssl.CERT_REQUIRED
 with socket.socket() as listener:
  listener.bind(('172.18.132.41',0));listener.listen(1);listener.settimeout(180)
  print('PORT',listener.getsockname()[1],flush=True)
  raw,peer=listener.accept()
  assert peer[0]=='172.18.33.60',peer
  sock=ctx.wrap_socket(raw,server_side=True)
sock.settimeout(180)
assert len(set(ids))==128
with sock, sock.makefile('rb') as inp:
 for k in ['sdxl','dmd2','weights']:
  for i in ([None] if k=='weights' else ids):
   if k=='weights':
    path=(base if send else root)/'models/baselines/dmd2/unet.safetensors'
   else:path=(root/('images' if send else 'reference')/k/f'{i:05d}.png')
   if send:
    with path.open('rb') as f:
     sock.sendall(struct.pack('>Q',path.stat().st_size))
     for data in iter(lambda:f.read(1<<20),b''):sock.sendall(data)
   else:
    size=struct.unpack('>Q',inp.read(8))[0]
    assert 0<size< (8<<30 if k=='weights' else 20<<20)
    tmp=path.with_suffix('.partial');path.parent.mkdir(exist_ok=True,parents=True);h=hashlib.sha256()
    with tmp.open('wb') as f:
     while size:
      data=inp.read(min(1<<20,size))
      if not data:raise EOFError()
      f.write(data);h.update(data);size-=len(data)
    if k=='weights':assert h.hexdigest()=='3549f1726d24afa4a26493c9049af8fb74108ff0a36f67ced0d3b4aa164cce92'
    tmp.replace(path)
 if not send:(root/'assets.ready').write_text('256 original PNGs and verified DMD2 weight\n')
print('COMPLETE',flush=True)
