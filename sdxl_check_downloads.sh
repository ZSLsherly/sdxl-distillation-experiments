source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
tail -n 8 "$root/logs/download.log"
tail -n 8 "$root/logs/data_download.log"
du -sh "$root/models" "$root/datasets"
python - <<'PY'
from pathlib import Path
import torch,pickle,io,numpy as np
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/datasets/dmd2')
class PlainUnpickler(pickle.Unpickler):
 def find_class(self,module,name):raise pickle.UnpicklingError('Unsupported pickle global '+module+'.'+name)
for name in ['captions_laion_score6.25.pkl','captions_coco14_test.pkl']:
 path=root/name
 if path.exists():
  value=PlainUnpickler(io.BytesIO(path.read_bytes())).load();print(name,type(value),len(value),str(value)[:300])
path=root/'vae_latents_000.pt'
if path.exists():
 with torch.serialization.safe_globals([np.core.multiarray._reconstruct,np.ndarray,np.dtype,type(np.dtype(np.float16)),type(np.dtype(np.float32)),type(np.dtype('U1'))]):
  data=torch.load(path,map_location='cpu',weights_only=True)
 print('LATENT_KEYS',{k:(type(v).__name__,getattr(v,'shape',None)) for k,v in data.items()})
 for k,v in data.items():
  if k!='latents':print(k,str(v)[:600])
 print('LATENTS_STATS',np.asarray(data['latents']).dtype,float(np.asarray(data['latents']).std()))
PY
