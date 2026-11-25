ROOT=/data5/shenglin.zeng
printf 'CLONE STATUS\n'
ps -u "$(id -un)" -o pid,etime,args | grep -E 'git clone|git-remote|index-pack' | head -10
ls -la "$ROOT/src/SparkDiffusion" | head
printf '\nEXISTING BASE PYTHON\n'
ls -l /data4/liushaozhen/miniforge3/bin/python /data6/liushaozhen/miniforge3/bin/python
printf '\nNETWORK DOWNLOAD CHECK\n'
curl -IL --connect-timeout 5 --max-time 10 -s https://hf-mirror.com | head -3
printf '\nLOCAL CUDA TOOLS\n'
ls -l /usr/local/cuda/bin/nvcc
