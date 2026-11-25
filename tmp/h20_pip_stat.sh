f=/root/autodl-tmp/h20_gan_mb32/blas_install.log
grep -Eo 'Collecting [^ ]+|Installing collected packages.*|Successfully installed.*|ERROR:.*|[0-9]+\.[0-9]+/[0-9]+\.[0-9]+ MB' "$f"|tail -4
find /root/autodl-tmp/sdxl_runs/tmp -name "*.whl" -printf "%s %f\n" | tail -6; df -h /root/autodl-tmp | tail -1; tail -c 300 /root/autodl-tmp/h20_gan_mb32/blas_install.log | sed -E "s@https?://[^ ]+@[index]@g"
