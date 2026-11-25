set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/dm1_mid4_uniform_mix1_g160/code"
cd "$c"
export PYTHONPATH="$c:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code:$SDXL_ROOT/code/sdxl_research"
grep -nE 'turning\.|turning_sampling|critic_stage\(|local_rollout\(|generated, final|pcm_loss\(|stage_labels|xt = |loss = 0.5|mix_shared|return 0.5|selected not in|k == 1' train.py boundary.py
sed -n '3,35p' run_quota_bc.py
grep -n 'mse\|mean\|fcount\|primary' train.py | tail -24
python - <<'PY'
import queue_dm_pair_coco as q,inspect
print(inspect.getsource(q.on_idle_gpu))
PY
ls "$SDXL_ROOT/outputs/dm1_mid4_uniform_mix1_g160"
tail -2 "$SDXL_ROOT/outputs/A_pcm10_g160/queue.log"

