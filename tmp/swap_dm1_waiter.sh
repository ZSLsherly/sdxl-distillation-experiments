set -eu
. /root/sdxl_research/env.sh
tar xzf "$SDXL_ROOT/tmp/dm1_review_deploy.tgz" -C "$SDXL_ROOT/tmp" dm1_uniform_reviewed.tgz
tar xOzf "$SDXL_ROOT/tmp/dm1_review_deploy.tgz" start_dm1_uniform_control.sh >"$SDXL_ROOT/tmp/start_dm1_uniform_reviewed.sh"
n=dm1_mid4_uniform_mix1_g160
o="$SDXL_ROOT/outputs/$n"; c="$SDXL_ROOT/tmp/$n/code"; check=$(mktemp -d "$SDXL_ROOT/tmp/dm1_review.XXXXXX")
cd "$check";tar xzf "$SDXL_ROOT/tmp/dm1_uniform_reviewed.tgz"
export PYTHONPATH="$check:$c:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code:$SDXL_ROOT/code/sdxl_research" OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES='' python audit_dm1_cpu.py
grep -o 'text=True)' "$SDXL_ROOT/tmp/c_mix1_first_dm_fake3_g160/code/launch.py" | wc -l | grep -x 2
# Only replace a verified sleeping waiter, while the upstream queue holds its lock.
exec 8>"$SDXL_ROOT/outputs/A_pcm10_g160/queue.lock"
if flock -n 8;then echo 'upstream finished; inspect live queue';exit 1;fi
p=$(cat "$o/queue.pid")
test "$p" = 309942
test "$(readlink -f "/proc/$p/cwd")" = "$(readlink -f "$c")"
tr '\0' ' ' <"/proc/$p/cmdline" | grep -q 'python -u run_dm1_uniform_control.py'
test -z "$(cat "/proc/$p/task/$p/children")"; test ! -e "$o/model";test ! -e "$o/train.log"
kill -TERM "$p"
flock -w 10 "$o/queue.lock" true
bash "$SDXL_ROOT/tmp/start_dm1_uniform_reviewed.sh"

