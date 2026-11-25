date '+%F %T %Z'
hostname
df -hP | awk 'NR==1 || $6=="/" || $6 ~ /^\/data[0-9]*$/'
for dir in /data5/shenglin.zeng/a50k_20261124/outputs/A50K_singlepass/training_state /data6/shenglin.zeng/a50k_20261124/no_middle/training_state; do
 if test -d "$dir"; then du -sh "$dir"; find "$dir" -mindepth 2 -maxdepth 2 -name complete.json -printf '%h\n'; fi
done
