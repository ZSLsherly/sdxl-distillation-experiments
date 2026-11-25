import hashlib,json,time,requests
from pathlib import Path
url='https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'
expected=url.split('/')[-2]
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/models/metrics');root.mkdir(parents=True,exist_ok=True)
target=root/'ViT-B-32.pt'
if target.exists():
    assert hashlib.sha256(target.read_bytes()).hexdigest()==expected
else:
    for attempt in range(3):
        try:
            session=requests.Session();session.trust_env=True
            with session.get(url,stream=True,timeout=(20,90)) as response:
                response.raise_for_status();digest=hashlib.sha256();total=0
                with target.with_suffix('.incomplete').open('wb') as stream:
                    for block in response.iter_content(1024**2):stream.write(block);digest.update(block);total+=len(block)
                assert digest.hexdigest()==expected,(digest.hexdigest(),total)
            target.with_suffix('.incomplete').replace(target);break
        except Exception as error:
            print('RETRY',attempt,type(error).__name__,flush=True)
            if attempt==2:raise
            time.sleep(2)
(root/'clip_manifest.json').write_text(json.dumps({'model':'OpenAI CLIP ViT-B/32','url':url,'sha256':expected,'bytes':target.stat().st_size,
    'purpose':'pilot held-out text alignment and seed diversity; not the DMD2 paper ViT-G/14 benchmark'},indent=2))
print('CLIP_VERIFIED',target.stat().st_size,flush=True)
