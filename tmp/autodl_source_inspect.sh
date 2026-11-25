set -eu
root=/data5/shenglin.zeng/sdxl_crossdistill
ls -lh "$root/exports/turning_20261026/environment.tar.gz" "$root/exports/turning_20261026/sdxl_sampling.tar" "$root/datasets/parallel_10k_seed20261018/prepared.pt"
tar -tzf "$root/exports/turning_20261026/environment.tar.gz" | head -n 12
find /data5/shenglin.zeng -maxdepth 3 -name 'python3.12' -o -name 'libpython3.12.so.1.0' 2>/dev/null
