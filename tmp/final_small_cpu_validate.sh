set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill
tar -xzf sdxl_research_20261022.tar.gz -C sdxl_research
cd sdxl_research
nice -n 10 python -m unittest discover -v 2>&1 | tee cpu_tests_after_gpu_review.log
python train.py --help >/dev/null
