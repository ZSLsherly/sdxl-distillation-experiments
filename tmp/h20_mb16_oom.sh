cd /root/autodl-tmp/h20_gan_mb16/code
grep -A1 -E '^\[rank0\].*/(modeling_sam|sam_rows|train|image_views|resnet|activations)\.py", line' h20.log | tail -21
grep -m1 'OutOfMemoryError: CUDA out of memory' h20.log | cut -c1-500

