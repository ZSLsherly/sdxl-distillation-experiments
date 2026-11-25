set -eu
r=/data5/shenglin.zeng/tools/filezilla
cd "$r/debs"
apt-get -qq download libfilezilla0 libpugixml1v5 libwxgtk3.0-gtk3-0v5 libwxbase3.0-0v5
for f in lib*.deb;do dpkg-deb -x "$f" "$r/app";done
export LD_LIBRARY_PATH="$r/app/usr/lib/x86_64-linux-gnu"
ldd "$r/app/usr/bin/filezilla" | grep 'not found' | sort -u || true
find "$r/app" -name '*sftp*'
find "$r/app" -name fzsftp -type f -exec sh -c '"$1" -V 2>&1 | head -4' sh {} ;
command -v ssh-keyscan
