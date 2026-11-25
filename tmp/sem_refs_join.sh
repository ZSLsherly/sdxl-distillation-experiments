set -eu
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_references_1024
w=models/baselines/dmd2/unet.partial
test "$(stat -c %s "$w")" = 5132533760
test "$(stat -c %s xfer/model_tail.bin)" = 2615976
cat xfer/model_tail.bin >> "$w"
test "$(sha256sum "$w" | cut -d' ' -f1)" = 3549f1726d24afa4a26493c9049af8fb74108ff0a36f67ced0d3b4aa164cce92
mv "$w" models/baselines/dmd2/unet.safetensors
test "$(find reference/sdxl -name '*.png'|wc -l)" = 128 && test "$(find reference/dmd2 -name '*.png'|wc -l)" = 128
echo '256 original PNGs and verified DMD2 weight' > assets.ready
cat assets.ready;ps -p 35090 -o pid,stat,etime,args;ls images/sdxl | wc -l
