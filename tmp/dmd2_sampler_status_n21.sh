date
ps -eo pid,etime,stat,pcpu,rss,args | grep '[p]ython.*check_native_sampler.py' || true
nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader
find /data1/shenglin.zeng/dmd2_matched_10k_20261119/runs/formal_g160_restart_20261120/native_sampler_check -maxdepth 1 -type f -printf '%f\n'
