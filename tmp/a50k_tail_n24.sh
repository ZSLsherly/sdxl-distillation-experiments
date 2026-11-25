R=/data/shenglin.zeng/a50k_20261123
cat "$R/records/prepare_status.json"
tail -12 "$R/records/build.log"
ls -lh "$R/datasets/parallel_50k_seed20261018.building/parts/selected.pt" 2>/dev/null || true
