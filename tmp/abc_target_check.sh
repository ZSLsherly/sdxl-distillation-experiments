set -eu
base=/data5/shenglin.zeng
test -d "$base" && test -w "$base"
df -B1 --output=source,avail,target "$base"
stat -c 'base %n %U %G %a' "$base"
if test -e "$base/checkpoints/a800_abc_g160_20261102"; then ls -lah "$base/checkpoints/a800_abc_g160_20261102"; fi
python3 - <<'PY'
import socket
try:
    with socket.create_connection(('connect.nma1.seetacloud.com',14892),5) as s:
        print('lab_to_rental_ssh',s.recv(96).decode().strip())
except OSError as e: print('lab_to_rental_ssh',repr(e))
PY

