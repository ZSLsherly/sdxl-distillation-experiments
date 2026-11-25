set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
sha256sum sdxl_parallel/train.py sdxl_parallel/training_state.py
bash -n sdxl_parallel/start_joint34_4gpu.sh
df -h /data5/shenglin.zeng
