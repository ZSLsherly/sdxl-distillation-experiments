set -eu
ROOT=/data5/shenglin.zeng
printf 'DIRECT DISK WRITE 2GiB\n'
dd if=/dev/zero of="$ROOT/benchmarks/disk-test.bin" bs=8M count=256 oflag=direct conv=fdatasync status=progress 2>&1
printf '\nDIRECT DISK READ 2GiB\n'
dd if="$ROOT/benchmarks/disk-test.bin" of=/dev/null bs=8M iflag=direct status=progress 2>&1
rm -- "$ROOT/benchmarks/disk-test.bin"
printf '\nDOWNLOAD PROGRESS\n'
tail -5 "$ROOT/logs/install_torch.log"
du -sh "$ROOT/src/SparkDiffusion/.git" "$ROOT/cache/pip" 2>/dev/null
ps -u "$(id -un)" -o pid,etime,args | grep -E 'git clone|pip install' | head -5
