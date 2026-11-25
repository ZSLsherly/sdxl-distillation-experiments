set -euo pipefail
base=/data3/shenglin.zeng
root="$base/sdxl_turning_ablation_20261028"
mkdir -p "$base"
test ! -e "$root"
mkdir -p "$root/transfers" "$root/tmp" "$root/cache" "$root/outputs"
df -h "$root" | tail -1
echo PREPARED="$root"
