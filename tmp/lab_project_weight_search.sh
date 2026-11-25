set -eu
find /data5/shenglin.zeng -xdev -type f -size 5135506134c ! -path '*/a800_archive_20261105/*' -printf '%p\n'
echo OTHER_ARCHIVES
find /data5/shenglin.zeng -xdev -type f -size +4G \( -iname '*.tar*' -o -iname '*.zip' \) -printf '%p %s\n'

