tail -n 45 /root/autodl-tmp/sdxl_research/logs/system_migration.log
cat /root/sdxl_research/storage_layout.json 2>/dev/null || true
ps -p 1987 -o pid,etime,stat,args || true
