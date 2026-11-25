set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT/src/SparkDiffusion"
python -m compileall -q sparkdiffusion imaginaire scripts
bash -n scripts/env.sh scripts/distill/*.sh
python -c 'from sparkdiffusion.configs.registry_distill import make_config; c=make_config(); print("CONFIG_IMPORT_OK")'
printf '\nDOWNLOAD\n'
tail -6 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
