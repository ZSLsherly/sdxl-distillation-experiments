set -eu
root=/data5/shenglin.zeng/tools/filezilla
mkdir -p "$root/debs" "$root/app"
cd "$root/debs"
python3 - <<'PY'
import socket,time
host='connect.nma1.seetacloud.com'
try:
 with socket.create_connection((host,14892),timeout=8) as s:
  print('A800 TCP connected',s.getpeername(),flush=True);s.settimeout(8);print('SSH banner',s.recv(64),flush=True)
except OSError as e:print('Direct SFTP probe failed:',e,flush=True)
PY
apt-get -qq -o Acquire::Retries=0 -o Acquire::http::Timeout=8 download filezilla filezilla-common || exit 19
for f in ./*.deb; do dpkg-deb -x "$f" "$root/app"; done
ldd "$root/app/usr/bin/filezilla" | grep 'not found' || true
