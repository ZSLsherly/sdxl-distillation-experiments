set -eu
. /root/sdxl_research/env.sh
j="$SDXL_ROOT/outputs/c_mix1_pcm_dm4_matched_time"
c="$SDXL_ROOT/tmp/c_mix1_pcm_dm4_matched_time/code"
cd "$c"
export PYTHONPATH="$c:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code:$SDXL_ROOT/code/sdxl_research" CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
nohup flock -n "$j/queue.lock" python -u - >>"$j/recover_eval.log" 2>&1 <<'PY' &
from run_pcm_dm4_comparison import job,read,evaluate_checkpoint,reference,save
model=job/'model'
p,=(model/'weights').glob('*_g000160_update_milestone/complete.json')
assert read(p)['student_updates']==160 and read(p)['fake_updates']==320
s=evaluate_checkpoint('PCM_DM4_G160',p.parent,read(model/'config.json'),read(reference/'manifest.json'),job)
save(job/'g160_scores.json',s)
print('G160_COMPLETE',s,flush=True)
PY
pid=$!
sleep 3
ps -p "$pid" -o pid,stat,comm || cat "$j/recover_eval.log"
