nvidia-smi -i 1,6 --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits
CUDA_VISIBLE_DEVICES=GPU-dee80cca-d29a-9086-d006-b76ba6ee093e /data1/shenglin.zeng/sdxl_eval_uniform_20261025/env/bin/python -c 'import torch;print(torch.cuda.is_available(),torch.cuda.device_count()); x=torch.ones(1,device="cuda"); print(x,torch.cuda.get_device_name())'
