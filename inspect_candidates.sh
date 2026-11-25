printf 'DATA USER DIRECTORIES\n'
for d in /data[1-7]; do printf '\n%s\n' "$d"; ls -1 "$d" 2>/dev/null; done
printf '\nCURRENT ACCOUNT ENVIRONMENTS\n'
ls -ld /data4/liushaozhen/miniforge3 /data4/liushaozhen/miniforge3/envs/* 2>/dev/null
printf '\nRELEVANT PROJECTS\n'
find /data[1-7] -maxdepth 3 -type d \( -iname '*shenglin*' -o -iname '*shaozhen*' -o -iname '*crossdistill*' -o -iname '*sparkdiffusion*' -o -iname '*wan*' -o -iname '*verl*' \) -print 2>/dev/null
printf '\nPACKAGE VERSIONS FROM METADATA\n'
/usr/bin/python3 - <<'PY'
import glob,json,os
for env in glob.glob('/data4/liushaozhen/miniforge3/envs/*'):
 print('\nENV',env)
 for name in ['python','pytorch','pytorch-cuda','cuda-version']:
  for f in glob.glob(env+'/conda-meta/'+name+'-*.json'):
   x=json.load(open(f))
   if x.get('name')==name: print(name,x.get('version'))
 for package in ['torch','triton','diffusers','transformers','flash_attn']:
  for f in glob.glob(env+'/lib/python*/site-packages/'+package+'-*.dist-info/METADATA'):
   for line in open(f):
    if line.startswith('Version:'): print(package,line.strip());break
PY
