cd /data/shenglin.zeng/a50k_20261123/eval_A_milestones_20261125 || exit 1
for i in 1 2 3 4 5 6 7 8 9;do
 n=$(find results -maxdepth 2 -name status_0.json | wc -l)
 if [ "$n" -eq 4 ];then break;fi
 if ! kill -0 18825 2>/dev/null;then break;fi
 sleep 5
done
for d in results/A*; do echo "$d"; grep -aE '^PROGRESS|^STRICT_STUDENT_LOADED' "$d/worker.log" | tail -1 | cut -c1-225;tail -c 270 "$d/worker.log" | tr '\r' '\n' |tail -1;done
