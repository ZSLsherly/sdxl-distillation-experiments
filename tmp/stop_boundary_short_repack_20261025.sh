set -eu
out=/data5/shenglin.zeng/sdxl_crossdistill/outputs/boundary_dmd_10k_a100_20261025_v1
printf 'Finish current cycle, then repack microbatch for the same 8-sample short test.\n' > "$out/short_retry1/STOP_REQUEST"
tail -10 "$out/short_retry1.log"
