set -eu
cd /root/autodl-tmp/sdxl_runs
df -h . /root
du -h --max-depth=1 . |sort -h
du -h --max-depth=1 outputs|sort -h
find -L outputs -name student_rank0.pt -printf '%p %s\n'
find . -type f -size +800M ! -name student_rank0.pt -printf '%p %s\n'

