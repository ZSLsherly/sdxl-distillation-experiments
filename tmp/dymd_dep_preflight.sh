set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
env -u http_proxy -u https_proxy "$B/env/bin/pip" install --dry-run -r "$B/official/pf_wan/requirements.txt" torch==2.7.1 torchvision==0.22.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/ --report "$B/dependency_resolution.json" > "$B/dependency_resolution.log" 2>&1 || tail -25 "$B/dependency_resolution.log"
tail -3 "$B/dependency_resolution.log"
