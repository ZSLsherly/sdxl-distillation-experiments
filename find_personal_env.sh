printf 'NAME MATCHES IN ACCOUNT DIRECTORIES\n'
find /home/liushaozhen /data3/liushaozhen /data4/liushaozhen /data6/liushaozhen \( -type d \( -name pkgs -o -name .git -o -name node_modules -o -name site-packages -o -name .cache \) -prune \) -o \( -iname '*shenglin*' -o -iname '*zeng*' -o -iname '*zsl*' \) -print 2>/dev/null | head -100
printf '\nUNREGISTERED VIRTUAL ENVIRONMENTS\n'
find /home/liushaozhen /data3/liushaozhen /data4/liushaozhen /data6/liushaozhen -maxdepth 7 \( -type d \( -name pkgs -o -name .git -o -name node_modules -o -name site-packages -o -name .cache \) -prune \) -o \( -name pyvenv.cfg -o -name conda-meta \) -print 2>/dev/null | head -100
printf '\nPERSONAL DIRECTORY NAMES ON DATA DISKS\n'
find /home /data[1-7] -maxdepth 3 -type d \( -iname '*shenglin*' -o -iname '*zeng*' -o -iname '*zsl*' \) -print 2>/dev/null | head -100
