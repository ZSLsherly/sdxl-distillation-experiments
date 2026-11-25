set -eu
. /root/sdxl_research/env.sh
command -v python
ls -ld /root/.ssh/authorized_keys2
readlink -f /root/.ssh; readlink -f /root/.ssh/authorized_keys2; stat -c "%U %a %F" /root/.ssh/authorized_keys2; namei -l /root/.ssh/authorized_keys2
