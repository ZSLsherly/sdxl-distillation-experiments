set -eu
umask 077
d=/data5/shenglin.zeng/sdxl_crossdistill/tmp/ref_sem_xfer_20261104
mkdir -p "$d";cd "$d";openssl req -x509 -newkey rsa:2048 -nodes -keyout local.key -out local.crt -days 1 -subj /CN=sem-send 2>/dev/null
