set -eu
. /root/sdxl_research/env.sh
cd /root/autodl-tmp/h20_C_mix1_formal_20261103
date
ps -p 13613 -o pid,etime,stat,args || true
tail -8 queue.log
for f in vae_compare.json selection.json;do if test -f "$f";then cat "$f";fi;done
for n in h20_fp32sam4_vae2_probe h20_C_mix1_terminal_GAN_G160;do
if test -f "$n.log";then tail -3 "$n.log" | cut -c1-400;fi
if test -f "$SDXL_ROOT/outputs/$n/status.json";then tail -c 800 "$SDXL_ROOT/outputs/$n/status.json";fi
if test -f "$SDXL_ROOT/outputs/$n/rank0.jsonl";then tail -1 "$SDXL_ROOT/outputs/$n/rank0.jsonl" | cut -c1-350;fi
done
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader


