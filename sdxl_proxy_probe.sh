source /data5/shenglin.zeng/env.sh
export HTTPS_PROXY=http://127.0.0.1:18797 HTTP_PROXY=http://127.0.0.1:18797
python - <<'PY'
import requests
r=requests.get('https://huggingface.co/api/models/stabilityai/stable-diffusion-xl-base-1.0',timeout=30);r.raise_for_status();print('PROXY_MODEL_API_OK',r.json()['sha'])
PY
