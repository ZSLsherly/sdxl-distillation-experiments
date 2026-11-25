printf 'EXISTING WAN MODEL DIRECTORIES\n'
find /data[1-7] /home -maxdepth 5 \( -type d \( -name pkgs -o -name node_modules -o -name .git -o -name site-packages \) -prune \) -o \( -type d \( -iname '*Wan2.1*1.3*' -o -iname '*Wan2.1*T2V*' -o -iname '*SparkWan*' \) -print \) 2>/dev/null | head -60
