set -eu
R=/data/shenglin.zeng/a50k_20261123
cd "$R/eval_A_milestones_20261125"
nohup bash -c '
set -eu
exec 9>sequential_read.lock
flock -n 9
for p in models/sdxl-base-1.0/{unet,text_encoder,text_encoder_2,vae}/*fp16.safetensors ../checkpoints/singlepass/samples*/student_rank*.pt; do
 if [ -f "$p" ]; then cat "$p" >/dev/null;fi
done
echo FINISHED
' >seqread.log 2>&1 </dev/null &
echo "CACHE_READER=$!"
