set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill/sdxl_research
python -m unittest -v test_runtime test_batching test_experiments 2>&1 | tee cpu_tests.log
python train.py --help > cli_help.txt
