root=/root/autodl-tmp/sdxl_research
date -Is
tail -n 18 "$root/logs/system_migration.log"
ps -p 1011 -o pid,etime,stat,args || true
df -h / /root/autodl-tmp
