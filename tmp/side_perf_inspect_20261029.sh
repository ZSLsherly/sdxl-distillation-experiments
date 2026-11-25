set -eu
nvidia-smi topo -m
nvidia-smi topo -p2p r
find /data1/shenglin.zeng/teacher_turning_20261026 -maxdepth 3 -name python -o -name model_index.json -o -name config.json
ls /data1/shenglin.zeng/teacher_turning_20261026
