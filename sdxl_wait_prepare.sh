set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
cd "$root"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
nohup bash -c '
set -eu
root=/data5/shenglin.zeng/sdxl_crossdistill
until [ -f "$root/datasets/dmd2/vae_latents_000.pt" ] && [ -f "$root/models/sdxl-base-1.0/tokenizer_2/vocab.json" ]; do sleep 10; done
exec python -u "$root/src/sdxl_prepare.py"
' > logs/prepare.log 2>&1 < /dev/null &
echo $! > logs/prepare.pid
cat logs/prepare.pid
