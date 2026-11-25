set -eu
cd /root/autodl-tmp/sdxl_runs
ls -la datasets
du -h --max-depth=2 tmp |sort -h |tail -15
find . -maxdepth 3 -type l -printf '%p -> %l\n' |head -65

