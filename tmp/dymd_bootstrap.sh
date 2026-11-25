set -eu
BASE=/data1/shenglin.zeng/dymd_prior_8_20261101
mkdir -p "$BASE"
export http_proxy=http://127.0.0.1:18897 https_proxy=http://127.0.0.1:18897
curl -sSL --connect-timeout 15 --max-time 40 https://huggingface.co/api/models/DAGroup-PKU/PF_Wan > "$BASE/model_api.json"
python3 - "$BASE/model_api.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));print('MODEL_SHA',d.get('sha')); print('\n'.join(x['rfilename'] for x in d.get('siblings',[])))
PY
if [ ! -d "$BASE/official/.git" ]; then git clone --depth 1 https://github.com/DAGroup-PKU/PhysisForcing.git "$BASE/official"; fi
git -C "$BASE/official" rev-parse HEAD
find "$BASE/official" -name AGENTS.md
