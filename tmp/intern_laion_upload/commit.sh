set -eu
work=/tmp/laion-discovery-904.b9IQDB
ctl="$HOME/.discovery/bin/discovery-ctl"
repo=Ashly/LAION-10K
"$ctl" --config-dir "$work/config" dataset commit --dataset "$repo" --message "Add LAION SDXL cache: 10000 training and 32 held-out samples" --format json
"$ctl" --config-dir "$work/config" dataset list --dataset "$repo" --version main --format json
