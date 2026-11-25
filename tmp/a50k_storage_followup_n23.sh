date
id
for r in /data /data1 /data2 /data3 /data4 /data5 /data6; do
 if [ -d "$r" ]; then
  echo "TOP=$r writable=$(test -w "$r" && echo yes || echo no)"
  find "$r" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null | head -45
 fi
done
for r in /data4/shenglin.zeng /data5/shenglin.zeng /data5/shenglin.zeng/checkpoints/a800_archive_20261106/datasets; do
 if [ -d "$r" ]; then
  echo "PERSONAL=$r"
  find "$r" -maxdepth 5 -type f \( -iname '*20k*' -o -iname '*50k*' -o -name '*manifest*.json' -o -name '*.tar*' \) -printf '%p %s\n' 2>/dev/null | head -70
 fi
done
