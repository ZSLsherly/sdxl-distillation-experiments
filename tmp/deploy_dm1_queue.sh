set -eu
. /root/sdxl_research/env.sh
name=c_mix1_first_dm_fake3_g160
c="$SDXL_ROOT/tmp/$name/code"; o="$SDXL_ROOT/outputs/$name"; setup="$SDXL_ROOT/tmp/$name/setup"
mkdir -p "$setup"
tar xzf "$SDXL_ROOT/tmp/first_dm_setup.tgz" -C "$setup"
python "$setup/prepare_dm1_ablation.py"
cd "$c"
export PYTHONPATH="$c" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' python "$setup/validate_dm1_queue.py"
export CUDA_VISIBLE_DEVICES=0
nohup python -u run_dm234_fake3.py >>"$o/queue.log" 2>&1 </dev/null &
echo $! > "$o/queue.pid"
sleep 2
kill -0 "$(cat "$o/queue.pid")"
cat "$o/queue.pid" "$o/queue.log"

