set -u
ls /data1/shenglin.zeng
ls /data1/shenglin.zeng/miniconda3/envs
find /data1/shenglin.zeng -maxdepth 5 -type d \( -iname '*wan*' -o -iname '*physis*' -o -iname '*runtime*' \) 2>/dev/null | head -50
for p in /data1/shenglin.zeng/miniconda3/bin/python /data1/shenglin.zeng/miniconda3/envs/*/bin/python; do
 "$p" -c 'import sys,importlib.util; print(sys.executable,sys.version.split()[0]); spec=importlib.util.find_spec("torch"); print(spec.origin if spec else "NO TORCH")' 2>/dev/null
done
for url in https://hf-mirror.com/api/models/DAGroup-PKU/PF_Wan https://huggingface.co/api/models/DAGroup-PKU/PF_Wan https://pypi.tuna.tsinghua.edu.cn/simple/; do
 curl -L -s --connect-timeout 8 --max-time 12 -o /dev/null -w '%{http_code} %{speed_download}\n' "$url"
done
ip -br addr
