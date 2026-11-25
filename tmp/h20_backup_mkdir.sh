set -eu
umask 077
b=/data5/shenglin.zeng/checkpoints/h20_gan_20261104
mkdir -p "$b"
test ! -f "$b/备份完成.txt"
