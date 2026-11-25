cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_joint4_uniform_1024
date -Is; cat run.log
ps -p 73690,80848 -o pid,stat,etime,cmd; for f in fetch_retry.log fetch.log gen*.log score.log;do if test -f "$f";then echo "$f";tail -5 "$f" | cut -c1-240;fi;done
du -h model/* 2>/dev/null | tail -9
echo IMAGES=$(find images -name '*.png' | wc -l)
ls -lh results.json 2>/dev/null || true

