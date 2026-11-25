set -eu
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_references_1024
date -Is
for a in sdxl dmd2;do
 echo "$a new=$(find "images/$a" -name '*.png' 2>/dev/null | wc -l) reused=$(find "reference/$a" -name '*.png' 2>/dev/null | wc -l)"
 if test -f "${a}_results.json";then ../../runtime/bin/python -c 'import json,sys;j=json.load(open(sys.argv[1]));print(j["n"],j["metrics"])' "${a}_results.json";fi
done
ls -lh assets.ready 2>/dev/null || true
tail -6 resume.log 2>/dev/null || true
ps -eo pid,stat,etime,args | grep -E 'generate.py (sdxl|dmd2)|run_references.py|score.py (sdxl|dmd2)|resume_after_workers.sh' | grep -v grep | cut -c1-160 || true
nvidia-smi --query-gpu=index,uuid --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader
