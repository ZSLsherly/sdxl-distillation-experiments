import requests,sys,json
from pathlib import Path
import importlib.util
spec=importlib.util.spec_from_file_location('upl','output/dymd_prior_8_20261101/upload_parallel.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rel='google/umt5-xxl/'
local=Path('output/dymd_prior_8_20261101/weights')/rel;local.mkdir(parents=True,exist_ok=True)
c=m.connect();s=c.open_sftp();remote=m.remote+'official/pf_wan/checkpoints/PF_Wan/'
for d in ['google','google/umt5-xxl']:
 try:s.mkdir(remote+d)
 except IOError:pass
for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','spiece.model']:
 r=requests.get('https://huggingface.co/DAGroup-PKU/PF_Wan/resolve/8fa7e17d5f944d0275df7dc7929137b956367ff8/'+rel+name,timeout=60);r.raise_for_status()
 p=local/name;p.write_bytes(r.content);m.upload(p,remote+rel+name)
 print('TOKENIZER_UPLOADED',name,len(r.content),flush=True)
s.close();c.close()
