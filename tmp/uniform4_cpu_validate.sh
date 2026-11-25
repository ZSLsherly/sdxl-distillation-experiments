set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill
mkdir -p sdxl_uniform4_20261022
tar -xzf uniform4_20261022.tar.gz -C sdxl_uniform4_20261022
cd sdxl_uniform4_20261022
python -m unittest -v test_schedule test_batching 2>&1 | tee cpu_tests.log
