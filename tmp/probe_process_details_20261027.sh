date -Is
root=/data5/shenglin.zeng/sdxl_crossdistill/outputs/turning_boundary_ablation_20261027
ps -eo pid,ppid,etime,stat,pcpu,rss,args | grep -E 'run_shared_probe.py|capped_turning_train.py' | grep -v grep
tail -8 "$root/short_normalized_u.log"
tail -3 "$root/shared_probe.log"
