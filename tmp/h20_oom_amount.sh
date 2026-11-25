cd /root/autodl-tmp/h20_gan_mb32/code
grep -m1 -o 'Tried to allocate[^.]*\.[0-9]* [GM]iB.* If reserved' blas126.log | cut -c1-500
grep -A1 'modeling_sam.py", line 803' blas126.log | head -2
