set -eu
cd /root/autodl-tmp/h20_gan_mb32/code
grep -A1 -E '^\[rank0\].*(modeling_sam.py|sam_pixel.py|OutOfMemoryError:)' blas126.log | tail -17
nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader
