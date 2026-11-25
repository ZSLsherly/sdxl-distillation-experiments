set -eu
date -Is
df -h / /root/autodl-tmp
du -x -h --max-depth=1 /root 2>/dev/null | sort -h
du -x -h --max-depth=1 /root/sdxl_research 2>/dev/null | sort -h
du -x -h --max-depth=1 /root/.cache /root/miniconda3/pkgs /tmp /var/cache /var/log 2>/dev/null | sort -h | tail -n 25
