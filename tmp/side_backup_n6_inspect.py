from backup_a800_nov5 import cloud,lab,run
with cloud() as s,lab() as d:
 print(run(s,"ps -eo pid,args | grep -E 'python|torchrun' | grep -v grep; df -h /root/autodl-tmp; cat /root/autodl-tmp/sdxl_runs/outputs/A_mid_clean_x0/comparison.json; cat /root/autodl-tmp/sdxl_runs/outputs/A_pcm10_mix1_g160/model/config.json"))
 print(run(d,"df -h /data5/shenglin.zeng; find /data5/shenglin.zeng -maxdepth 4 -iname '*20k*'"))
