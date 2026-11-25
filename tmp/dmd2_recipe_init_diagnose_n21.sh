R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
ps -eo pid,ppid,state,etime,pcpu,rss,wchan:24,args | grep -E '[t]rain_matched.py|[r]un_guard.py' || true
cat /sys/fs/cgroup/user.slice/memory.current
head -5 /sys/fs/cgroup/user.slice/memory.stat
cat /sys/fs/cgroup/user.slice/memory.events
find "$R/runs" -maxdepth 2 -type f -name 'precision_rank*.json' -print 2>/dev/null
grep -o '"dropout"[^,]*' /data1/shenglin.zeng/sdxl_eval_uniform_20261025/models/sdxl-base-1.0/unet/config.json
