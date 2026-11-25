R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
grep -B 14 -A 3 'out of memory' "$R/logs/smoke_micro4.log" | tail -26
ps -eo pid,args | grep -E '[t]rain_matched.py|[r]un_guard.py' || true
