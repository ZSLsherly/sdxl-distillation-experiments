source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
which python
python -c 'import torch,diffusers,transformers,accelerate,numpy,PIL; print(torch.__version__,diffusers.__version__,transformers.__version__,accelerate.__version__,numpy.__version__,PIL.__version__)'
du -sh /data5/shenglin.zeng/envs/* models/sdxl-base-1.0 models/dinov2-small datasets/parallel_10k_seed20261018 2>/dev/null
find models/sdxl-base-1.0 -maxdepth 2 -type f -printf '%P %s\n'
cat /data5/shenglin.zeng/env.sh
command -v rsync sshpass
