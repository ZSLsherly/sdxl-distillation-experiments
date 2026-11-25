set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
mkdir -p "$SDXL_WORK/datasets/20k_sources"
cd "$SDXL_WORK/datasets/20k_sources"
nohup flock -n download.lock nice -n 10 python -u "$SDXL_ROOT/tmp/fetch_20k_sources.py" > download.log 2>&1 </dev/null &
echo $!
