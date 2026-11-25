ROOT=/data5/shenglin.zeng
tail -6 "$ROOT/logs/install_torch.log"
tail -8 "$ROOT/logs/install_deps.log"
printf '\nMODEL FILE SIZES\n'
find "$ROOT/models" -type f -printf '%s %p\n' | sort -nr | head -8
printf '\nDISK ACTIVITY\n'
vmstat 1 3
