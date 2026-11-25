set -eu
work=/tmp/laion-discovery-904.b9IQDB
data=/data5/shenglin.zeng/sdxl_crossdistill/datasets/parallel_10k_seed20261018
"$HOME/.discovery/bin/discovery-ctl" --config-dir "$work/config" dataset upload-file --dataset Ashly/LAION-10K --local "$data/prepared.pt,$data/manifest.json" --remote-path / --format json >"$work/upload.log" 2>&1 || { tail -c 1700 "$work/upload.log"; exit 1; }
tail -c 2200 "$work/upload.log"
