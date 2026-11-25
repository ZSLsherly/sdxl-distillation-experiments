set -eu
OLD=/data1/shenglin.zeng/dmd2_matched_10k_20261119
R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
mkdir -p "$R/logs" "$R/tmp" "$R/cache"
if [ ! -e "$R/env" ]; then ln -s "$OLD/env" "$R/env"; fi
df -h /data1
free -h
cat /sys/fs/cgroup/user.slice/memory.current 2>/dev/null || true
cat /sys/fs/cgroup/user.slice/memory.high 2>/dev/null || true
ps -eo pid,args | grep -E '[t]rain_matched.py|[r]un_guard.py' || true
