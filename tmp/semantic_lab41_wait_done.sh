cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_joint4_uniform_1024
for i in $(seq 1 45);do if test -s results.json || ! kill -0 130015 2>/dev/null || grep -q Traceback score.log;then break;fi; sleep 4;done
tail -6 score.log
ls -l results.json 2>/dev/null || true
