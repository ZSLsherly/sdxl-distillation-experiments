cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_references_1024
date -Is;ls -l assets.ready
ps -eo pid,stat,etime,pcpu,pmem,args | grep -E '(generate.py|run_references.py|score.py)' | grep -v grep | head -8
tail -c 1000 sdxl_0.log | tr '\r' '\n' | tail -5
ls -t images/sdxl | head -1 | xargs -I@ stat -c 'latest_png %y' images/sdxl/@
timeout -k 1 4 nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader || true
