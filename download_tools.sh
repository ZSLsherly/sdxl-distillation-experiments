set -eu
ROOT=/data5/shenglin.zeng
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip" XDG_CACHE_HOME="$ROOT/cache" HF_HOME="$ROOT/cache/huggingface" PYTHONDONTWRITEBYTECODE=1
nohup /data4/liushaozhen/miniforge3/bin/python -m pip install --target "$ROOT/envs/download-tools" 'huggingface_hub<1' -i https://pypi.org/simple > "$ROOT/logs/download-tools.log" 2>&1 < /dev/null &
printf 'DOWNLOAD_TOOLS_PID=%s\n' "$!"
printf '\nREPO DOWNLOAD SIZE\n'
du -sh "$ROOT/src/SparkDiffusion/.git"
printf '\nHF MIRROR API\n'
curl -s --connect-timeout 5 --max-time 15 https://hf-mirror.com/api/models/Wan-AI/Wan2.1-T2V-1.3B | /usr/bin/python3 -c 'import sys,json; x=json.load(sys.stdin);print(x.get("id"));print("\n".join(f["rfilename"] for f in x.get("siblings",[])))'
