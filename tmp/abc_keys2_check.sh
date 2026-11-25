set -eu
if test -e /root/.ssh/authorized_keys2; then echo 'auxiliary key file already present'; else echo 'auxiliary key file unused'; fi
