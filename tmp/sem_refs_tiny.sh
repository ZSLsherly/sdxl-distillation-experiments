cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_references_1024 || exit 1
date -Is
for a in sdxl dmd2; do ls -1 "images/$a" 2>/dev/null | wc -l;ls -l "${a}_results.json" 2>/dev/null;done
tail -4 resume.log
