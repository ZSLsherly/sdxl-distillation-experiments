root=/root/autodl-tmp/sdxl_research
tail -n 8 "$root/logs/validate_lowmem.log"
cat /sys/fs/cgroup/memory.current
