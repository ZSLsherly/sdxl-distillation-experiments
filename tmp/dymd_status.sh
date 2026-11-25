B=/data1/shenglin.zeng/dymd_prior_8_20261101
tail -8 "$B/install.log"
tail -8 "$B/download.log"
du -sh "$B/official/pf_wan/checkpoints/PF_Wan" 2>/dev/null
ls "$B"/*READY 2>/dev/null || true
ps -p 2746444,2746445 -o pid,etime,args
