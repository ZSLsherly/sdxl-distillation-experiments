set -eu
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_references_1024
w=models/baselines/dmd2/unet.partial;stat -c %s "$w"; sha256sum "$w"
cat run_references.py | grep 'not in busy'
ps -eo args |grep -E '^.*python -u (generate|run_references).py'|grep -v grep || true
