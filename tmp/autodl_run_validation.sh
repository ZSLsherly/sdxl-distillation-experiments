set -eu
cd /root/autodl-tmp/sdxl_research
. ./env.sh
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -u tmp/validate_lowmem.py > logs/validate_lowmem.log 2>&1 || { tail -n 40 logs/validate_lowmem.log; exit 1; }
tail -n 65 logs/validate_lowmem.log
