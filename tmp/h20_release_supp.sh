set -eu
cd /root/autodl-tmp
find h20_gan_mb16 h20_gan_mb32 \( -type d \( -name blas126 -o -name __pycache__ -o -name '*.dist-info' -o -name '.git' \) -prune \) -o \( -type f \( -name '*.py' -o -name '*.sh' -o -name '*.json' -o -name '*.log' -o -name '*.csv' -o -name '*.txt' \) -size -32M -print0 \) | tar -czf h20_benchmark_sources_release.tar.gz --null -T -
sha256sum h20_benchmark_sources_release.tar.gz
du -h h20_benchmark_sources_release.tar.gz
