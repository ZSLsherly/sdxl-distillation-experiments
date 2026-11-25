df -h /data{3,4,5,6,7}
for p in /data{3,4,5,6,7}; do if test -w "$p"; then echo "$p writable"; fi; done
cd /data5/shenglin.zeng/a50k_20261124
cat outputs/A50K_singlepass/status.json | head -10
ls -ld outputs/A50K_no_middle* code/after_singlepass* 2>/dev/null || true
