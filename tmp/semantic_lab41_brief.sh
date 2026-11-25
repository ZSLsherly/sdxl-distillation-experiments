cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_joint4_uniform_1024
date -Is
echo IMAGES=$(find images -name '*.png' | wc -l)
for f in gen*.log;do echo "$f"; grep -aE 'Error|complete|Traceback' "$f" | tail -2;done
tail -2 score.log 2>/dev/null || true
ls -l results.json 2>/dev/null || true
stat -c 'public_model_upload_bytes=%s' /data1/shenglin.zeng/public_clip_B32_20261104.tar
