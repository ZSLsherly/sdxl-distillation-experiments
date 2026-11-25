id
printf '\nHOME AND STORAGE\n'
ls -ld "$HOME" /home/* /data /data[0-9]* /mnt/* /opt/conda /opt/anaconda* 2>/dev/null
printf '\nCURRENT ENVIRONMENT\n'
command -v conda python python3 uv
printf '\nREGISTERED ENVIRONMENTS\n'
for f in /home/*/.conda/environments.txt; do
  if [ -r "$f" ]; then printf '\n%s\n' "$f"; cat "$f"; fi
done
printf '\nENVIRONMENT AND PROJECT DIRECTORIES\n'
find /home /data /data1 /data2 /data3 /data4 /data5 /data6 /opt -maxdepth 4 -type d \( -iname '*conda*' -o -name envs -o -name .venv -o -iname '*shenglin*' -o -iname '*shaozhen*' -o -iname '*crossdistill*' -o -iname '*sparkdiffusion*' -o -iname '*wan*' -o -iname '*verl*' \) -print 2>/dev/null | head -250
