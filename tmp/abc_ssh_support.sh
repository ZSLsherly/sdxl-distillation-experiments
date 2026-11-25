set -eu
printf "port 22 SSH service: "
(command -v sshd || true)
test -d /root/.ssh && stat -c 'ssh directory %U %a' /root/.ssh || true
# Check supported effective settings without reading credentials or keys.
sshd -T 2>/dev/null | awk '$1=="pubkeyauthentication"||$1=="authorizedkeysfile"'
test -x /usr/bin/timeout
