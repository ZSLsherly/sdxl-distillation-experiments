set -eu
source /data5/shenglin.zeng/env.sh
mkdir -p "$CROSSDISTILL_ROOT/image_crossdistill"/{src,logs,outputs,datasets,models}
python - <<'PY'
import inspect,json,requests
from pathlib import Path
from diffusers import SanaPipeline,SanaTransformer2DModel
repo='Efficient-Large-Model/Sana_600M_512px_diffusers'
api=requests.get('https://hf-mirror.com/api/models/'+repo,timeout=30);api.raise_for_status();info=api.json()
print('REVISION',info['sha'])
print('FILES',[x['rfilename'] for x in info['siblings']])
for f in ['model_index.json','scheduler/scheduler_config.json','transformer/config.json','vae/config.json']:
 r=requests.get('https://hf-mirror.com/'+repo+'/resolve/'+info['sha']+'/'+f,timeout=30);r.raise_for_status();print(f,r.text)
Path('/data5/shenglin.zeng/image_crossdistill/model_info.json').write_text(json.dumps(info,indent=2))
print('FORWARD',inspect.signature(SanaTransformer2DModel.forward))
print('ENCODE',inspect.signature(SanaPipeline.encode_prompt))
print('PIPELINE',inspect.getfile(SanaPipeline))
print('FORWARD SOURCE',inspect.getsource(SanaTransformer2DModel.forward))
PY
