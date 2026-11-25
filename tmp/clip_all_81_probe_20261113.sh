df -h /data3
find /data3/shenglin.zeng -maxdepth 5 -type f \( -name model_index.json -o -name revision.txt -o -name pyvenv.cfg \) | head -30
ls -d /data3/shenglin.zeng/*/env/bin/python /data3/shenglin.zeng/*/runtime/bin/python* 2>/dev/null
nvidia-smi -i 5 --query-compute-apps=pid --format=csv,noheader,nounits
CUDA_VISIBLE_DEVICES=GPU-5eceb5c0-44bb-9b56-d40e-2846ce7ff00f /data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python -c 'import torch; print(torch.cuda.is_available());print(torch.ones(1,device="cuda"),torch.cuda.get_device_name())'
