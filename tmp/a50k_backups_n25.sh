cd /data/shenglin.zeng/a50k_20261123/checkpoints || exit 1
for dir in singlepass no_middle_singlepass; do echo "$dir";find "$dir" -maxdepth 2 -name complete.json -printf '%p\n' 2>/dev/null;done
