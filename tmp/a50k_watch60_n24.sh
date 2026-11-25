R=/data5/shenglin.zeng/a50k_20261124
cd "$R"
tail -n 3 logs/queue.log 2>/dev/null || true
tail -n 5 logs/restore_test.log 2>/dev/null || true
ls -lh datasets/parallel_50k_seed20261018/prepared.pt* datasets/*txt 2>/dev/null || true
for f in logs/first_update_mb*.log logs/A50K_G*.log; do
 if [ -f "$f" ]; then echo "$f"; tail -n 2 "$f" | cut -c1-350; grep -Ei 'out of memory|Error:|RESUMED' "$f" | tail -n 2 | cut -c1-350 || true; fi
done
for f in outputs/*/status.json; do if [ -e "$f" ];then head -c650 "$f"; echo;fi; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
