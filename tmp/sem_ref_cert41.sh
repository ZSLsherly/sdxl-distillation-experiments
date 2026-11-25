set -eu
umask 077
d=/data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_references_1024/xfer
mkdir -p "$d";cd "$d";openssl req -x509 -newkey rsa:2048 -nodes -keyout local.key -out local.crt -days 1 -subj /CN=sem-recv 2>/dev/null
