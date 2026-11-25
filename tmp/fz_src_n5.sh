set -eu
cd /data5/shenglin.zeng/tools/filezilla
f=$(apt-cache show filezilla | awk '/^Filename:/{print $2;exit}')
f="$(dirname "$f")/filezilla_3.46.3.orig.tar.bz2"
wget -q -T 15 "http://archive.ubuntu.com/ubuntu/$f" -O protocol.tar.bz2
mkdir -p src
tar xjf protocol.tar.bz2 -C src --wildcards '*/src/putty/*' '*/src/engine/sftp/*'
grep -r -n -A14 'ReadQuotas' src | head -60
