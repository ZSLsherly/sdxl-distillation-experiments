source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
tail -n 8 "$root/logs/teacher_stages.log"
if [ -f "$root/logs/teacher_stages.exit" ]; then cat "$root/logs/teacher_stages.exit";fi
python -c 'import importlib.util; print({k:bool(importlib.util.find_spec(k)) for k in ["matplotlib","PIL","clip"]})'
