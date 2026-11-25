set -eu
cd /data/shenglin.zeng/a50k_20261123/eval_A_milestones_20261125
if test -f code/queue.py; then mv -n code/queue.py code/run_milestones.py; fi
nohup /data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python -u code/run_milestones.py >>queue.log 2>&1 </dev/null &
echo "EVAL_QUEUE=$!"
