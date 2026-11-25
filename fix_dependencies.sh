set -eu
source /data5/shenglin.zeng/env.sh
python -m pip install pytz --disable-pip-version-check -q
cd "$CROSSDISTILL_ROOT/src/SparkDiffusion"
python -c 'from sparkdiffusion.configs.registry_distill import make_config; c=make_config(); print("CONFIG_IMPORT_OK")'
tail -6 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
