cd /root/autodl-tmp/h20_gan_mb32/code
grep -A45 -E 'Fatal Python|Floating point exception|H20_OOM|OutOfMemory' diag.log | head -49
tail -6 diag.log | cut -c1-250
command -v gdb || true
