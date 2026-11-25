from pathlib import Path
import concurrent.futures,hashlib,os,time,requests
work=Path(os.environ['SDXL_WORK'])/'datasets/20k_sources';work.mkdir(parents=True,exist_ok=True)
rev='be22767697a1f3ca656b73c776e15fa335c86c6c'
files=[('vae_latents_002.pt',1093248651,'dd26be16f0d5c8067fdefae8909434cb65008232d4463b9cb4be509318df51ba'),
       ('vae_latents_003.pt',1097920651,'e7ce92fdc7d3a7be2379e5bad082ff70265d7f23836ab4b22f36b0ce29deb6ad')]
def fetch(spec):
    name,size,digest=spec
    path=work/name; part=path.with_suffix('.partial')
    def valid(p):
        if not p.exists() or p.stat().st_size!=size:return False
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()==digest
    if path.exists():
        assert valid(path);return
    for a in range(5):
        if valid(part):part.replace(path);print('VERIFIED',name,flush=True);return
        start=part.stat().st_size if part.exists() else 0
        if start>=size:raise ValueError('Invalid partial '+name)
        url=f'https://hf-mirror.com/tianweiy/DMD2/resolve/{rev}/data/laion_vae_latents/sdxl_vae_latents_laion_500k/{name}?download=true'
        try:
            with requests.get(url,headers={'Range':f'bytes={start}-'} if start else {},timeout=(15,90),stream=True) as r:
                r.raise_for_status()
                if r.status_code==206:
                    assert r.headers['Content-Range']==f'bytes {start}-{size-1}/{size}'; mode='ab'
                else:assert r.status_code==200;mode='wb'
                with part.open(mode) as f:
                    for chunk in r.iter_content(4*1024**2):f.write(chunk)
        except (requests.RequestException,AssertionError) as e:print('retry',name,type(e).__name__,flush=True);time.sleep(3)
    if not valid(part):raise ValueError('Download incomplete '+name)
    part.replace(path)
with concurrent.futures.ThreadPoolExecutor(2) as pool:list(pool.map(fetch,files))
