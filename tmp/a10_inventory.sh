set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT"
ls outputs/turning_BC_paired_quota_20261102
ls outputs/turning_BC_paired_quota_20261102/uniform_control/source
cat code/sdxl_research/run_uniform_after_bc.py | head -95
ls outputs/bc_winner_late_mix_20261102
ps -eo pid,ppid,comm,args | awk '$3=="python" && /queue|after_|evaluate.py/ {print substr($0,1,210)}'
df -h outputs

