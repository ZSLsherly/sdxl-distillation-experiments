for py in /data1/shenglin.zeng/miniconda3/envs/sherly/bin/python /data1/shenglin.zeng/miniconda3/envs/temp/bin/python; do
  echo "$py"
  CUDA_VISIBLE_DEVICES='' "$py" -c 'import sys; print(sys.version); import torch; print("torch",torch.__version__); import diffusers,transformers; print("diffusers",diffusers.__version__,"transformers",transformers.__version__)'
done
