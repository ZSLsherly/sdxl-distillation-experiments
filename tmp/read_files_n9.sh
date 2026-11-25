set -eu
date -Iseconds
df -h /root /root/autodl-tmp
printf '\n== directory sizes ==\n'
du -xh -d 1 /root/autodl-tmp/sdxl_runs | sort -hr | head -20
printf '\n== files >=512 MiB, bytes ==\n'
find /root/autodl-tmp/sdxl_runs -type f -size +511M -printf '%s %p\n' | sort -nr | head -45
printf '\n== system ==\n'
du -xh -d 1 /root 2>/dev/null | sort -hr | head -18
