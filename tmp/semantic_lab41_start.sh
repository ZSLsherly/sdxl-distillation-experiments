set -eu
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs
mkdir semantic_joint4_uniform_1024
cd semantic_joint4_uniform_1024
tar -xzf /data1/shenglin.zeng/lab_sem_old.tgz
nohup bash launch.sh >run.log 2>&1 </dev/null & echo PID=$!
