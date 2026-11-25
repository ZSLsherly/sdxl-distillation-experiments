set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT"
df -B1 /root outputs
ls outputs/A_pcm10_g160; tail -3 outputs/A_pcm10_g160/queue.log
cat outputs/c_mix1_first_dm_fake3_g160/effective_config.json
grep -n 'turning_sampling\|def phase_random\|time_range=ranges' tmp/c_mix1_first_dm_fake3_g160/code/train.py |head -18
ls outputs/c_mix1_first_dm_fake3_g160/model/source
du -sh outputs/c_mix1_dm_pair_coco10k/DM1_mid4_fake3

