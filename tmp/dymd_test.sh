set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
/data1/shenglin.zeng/miniconda3/envs/sherly/bin/python "$B/test_measurement.py"
python3 "$B/prepare_run.py"
python3 "$B/progress.py"
