B=/data1/shenglin.zeng/dymd_prior_8_20261101
ps -p 4069690 -o pid,stat,etime,time,%cpu,%mem,args
cat /proc/4069690/io 2>/dev/null || true
ls -l --time-style=long-iso "$B/official/pf_wan/checkpoints/PF_Wan/"*.parallel 2>/dev/null
cat "$B/status.json"
