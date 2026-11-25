set -eu
root=/data/shenglin.zeng/a50k_20261123
mkdir -p "$root/datasets" "$root/checkpoints" "$root/records"
probe=$(mktemp "$root/records/write_probe.XXXXXX")
printf 'A50K storage write verification\n' > "$probe"
cat "$probe"
rm -- "$probe"
df -h "$root"
printf 'STORAGE_READY=%s\n' "$root"
