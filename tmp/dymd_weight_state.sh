B=/data1/shenglin.zeng/dymd_prior_8_20261101
ls -lh "$B/official/pf_wan/checkpoints/PF_Wan/"*.pth "$B/official/pf_wan/checkpoints/PF_Wan/"*.parallel 2>/dev/null
ps -u shenglin.zeng -o pid,etime,args | grep -E 'sha256sum|dymd_prior_8_20261101/controller|sftp-server' | head -25
