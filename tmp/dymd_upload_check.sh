B=/data1/shenglin.zeng/dymd_prior_8_20261101
du -h "$B/official/pf_wan/checkpoints/PF_Wan/Wan2.1_VAE.pth.parallel" "$B"/wheels/*.parallel
ls -l --time-style=long-iso "$B/official/pf_wan/checkpoints/PF_Wan/Wan2.1_VAE.pth.parallel" "$B"/wheels/*.parallel
ps -u shenglin.zeng -o pid,ppid,etime,args | grep -E 'sshd:|sftp' | head -20
