ls -l /data3/shenglin.zeng/sdxl_turning_ablation_20261028
find /data3/shenglin.zeng/sdxl_turning_ablation_20261028 -maxdepth 5 -name '*model*' | head -25
/data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python -c 'import torch,transformers,diffusers; print(torch.__version__,transformers.__version__,diffusers.__version__);import cleanfid'
