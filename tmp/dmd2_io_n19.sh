R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
for p in 455398 455399 455400 455401; do echo "PID=$p"; cat /proc/$p/io; done
df -h /data1
ps -p 455398,455399,455400,455401 -o pid,stat,pcpu,rss,wchan:24
ls -lh /data1/shenglin.zeng/sdxl_eval_uniform_20261025/datasets/parallel_10k_seed20261018/prepared.pt
