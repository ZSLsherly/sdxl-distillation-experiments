set -eu
. /root/sdxl_research/env.sh
df -h "$SDXL_ROOT/outputs"
cd "$SDXL_ROOT/outputs"
for d in c_mix1_dm234_fake3_20261103 c_mix1_first_dm_fake3_g160 c_mix1_dm_pair_coco10k; do echo "$d"; tail -3 "$d/queue.log"; test ! -f "$d/queue.pid" || ps -p "$(cat "$d/queue.pid")" -o pid,stat,comm; done
cat c_mix1_dm_pair_coco10k/baseline.json
cat bc_winner_late_mix_20261102/effective_config.json
cd "$SDXL_ROOT/tmp/c_mix1_dm234_fake3_20261103/code"
ls *.py

