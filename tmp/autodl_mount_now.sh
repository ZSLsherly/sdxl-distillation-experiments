date -Is
findmnt -T /root/autodl-tmp/sdxl_research -o TARGET,SOURCE,FSTYPE,AVAIL
findmnt -T /root -o TARGET,SOURCE,FSTYPE,AVAIL
ls -ld /root/autodl-tmp/sdxl_research/{env,runtime,models,datasets,code,transfers}
cat /root/autodl-tmp/sdxl_research/setup.status
