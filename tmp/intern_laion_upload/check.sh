set -eu
data=/data5/shenglin.zeng/sdxl_crossdistill/datasets/parallel_10k_seed20261018
stat -c '%n %s' "$data/prepared.pt" "$data/manifest.json"
python3 -c 'import json,sys; m=json.load(open(sys.argv[1])); print(m["train_count"],m["holdout_count"],m["seed"]); assert (m["train_count"],m["holdout_count"])==(10000,32)' "$data/manifest.json"
command -v discovery-ctl || true
test ! -x "$HOME/.discovery/bin/discovery-ctl" || "$HOME/.discovery/bin/discovery-ctl" version
curl -fsSI -m 10 https://pjlab-shanghai-moce.oss-cn-shanghai.aliyuncs.com/static/installpack/discovery-ctl/prod/v1.2.1/install.sh | head -3
