d=/root/autodl-tmp/h20_C_mix1_formal_20261103
cd "$d"
ps -eo pid,ppid,etime,stat,pcpu,rss,wchan:20,comm | awk '$2==11397 || $1==11397'
tail -10 queue.log
ls -l vae_compare.json selection.json 2>/dev/null || true
ls -d /root/sdxl_research/tmp/bc_public_reference.tar /root/sdxl_research/models/sdxl-base-1.0/* /root/autodl-tmp/sdxl_runs/bc_texture_inputs 2>/dev/null || true
