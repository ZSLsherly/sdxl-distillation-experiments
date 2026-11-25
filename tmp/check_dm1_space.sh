set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/tmp/c_mix1_dm234_fake3_20261103/code"
df -h "$SDXL_ROOT/outputs" /root "$SDXL_WORK"
readlink -f "$SDXL_ROOT/outputs"
ls -l "$SDXL_ROOT/outputs/c_mix1_dm234_fake3_20261103"
sha256sum train.py boundary.py
cat "$SDXL_ROOT/outputs/c_mix1_dm234_fake3_20261103/effective_config.json"
rg='' 
grep -nE 'fake_updates|both_middle|check.*probe|stage_labels|G160|320|480' train.py run_quota_bc.py | tail -45

