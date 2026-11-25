from pathlib import Path
p=Path('tmp/a50k_fetch_missing_direct_n23.py')
s=p.read_text(encoding='utf-8')
start=s.index(" path=folder/spec['name'];part=path.with_suffix('.partial')")
end=s.index(" assert path.stat().st_size==spec['size']",start)
s=s[:start]+""" path=folder/spec['name']
 for round_ in range(3):
  if path.exists():break
  try:download_parallel(spec,path)
  except RuntimeError:
   if round_==2:raise
   print('RESUME_INCOMPLETE_CHUNKS',spec['name'],flush=True);time.sleep(15)
"""+s[end:]
s=s.replace("download_mirror=url if not path.exists() else 'hf-mirror.com'","download_mirror='hf-mirror.com'")
s=s.replace("opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))\n","")
p.write_text(s,encoding='utf-8')
compile(s,str(p),'exec')
print('Removed obsolete sequential branch; retained chunk resume and SHA256 validation')
