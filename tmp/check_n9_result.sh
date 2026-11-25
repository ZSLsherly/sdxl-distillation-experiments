p=/root/autodl-tmp/sdxl_runs/outputs/turning_BC_masked_20261101
find "$p" -type f -name '*.png' | wc -l
ps -eo etime,comm,args | grep -E 'hashlib|[t]ar --null' | cut -c1-250 | head
which python || true
