date
R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
tail -n 5 "$R/logs/smoke_micro4.log"
for p in 2492557 2492558; do
  if [ -r /proc/$p/io ]; then echo "OWN_PID=$p"; grep -E 'read_bytes|write_bytes' /proc/$p/io; fi
done
grep 'safetensors' /proc/2492557/maps 2>/dev/null | head -3 || true
ps -p 2492557,2492558 -o pid,state,etime,pcpu,rss,wchan:24
cat /sys/fs/cgroup/user.slice/io.max 2>/dev/null || true
cat /sys/fs/cgroup/user.slice/memory.events
ls "$R/runs/smoke_micro4" 2>/dev/null || true
