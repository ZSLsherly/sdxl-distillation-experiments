date
ps -eo pid,etime,pcpu,rss,args | grep -E '[t]est_recipe.py|[t]est_cpu.py|[t]rain_matched.py' || true
cat /sys/fs/cgroup/user.slice/memory.events 2>/dev/null || true
cat /sys/fs/cgroup/user.slice/memory.stat 2>/dev/null | head -8
ls -l /data1/shenglin.zeng/dmd2_official_recipe_20261121/source_manifest.json 2>/dev/null || true
