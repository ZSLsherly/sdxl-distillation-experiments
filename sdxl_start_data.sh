set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT/sdxl_crossdistill"
export HTTPS_PROXY=http://127.0.0.1:18797 HTTP_PROXY=http://127.0.0.1:18797
export HF_ENDPOINT=https://huggingface.co
nohup python -u src/sdxl_download_data.py > logs/data_download.log 2>&1 < /dev/null &
echo $! > logs/data_download.pid
cat logs/data_download.pid
python - <<'PY'
import inspect
from diffusers import StableDiffusionXLPipeline
print('ENCODE',inspect.signature(StableDiffusionXLPipeline.encode_prompt))
print('TIMEIDS',inspect.signature(StableDiffusionXLPipeline._get_add_time_ids))
PY
