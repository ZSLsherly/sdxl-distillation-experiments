ROOT=/data5/shenglin.zeng
printf 'COMMUNICATION\n'; tail -12 "$ROOT/logs/gpu_comm_host.log"
printf '\nDEPENDENCIES\n'; tail -6 "$ROOT/logs/install_deps.log"
printf '\nMODELS\n'; du -sh "$ROOT/models"; tail -5 "$ROOT/logs/download_models.log"
printf '\nGPU PROCESS STATE\n'; ps -u "$(id -un)" -o pid,stat,etime,args | grep -E 'gpu_comm|find /data' | head -12
