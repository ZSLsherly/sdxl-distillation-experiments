id
findmnt /data4
ls -la /data4
for r in /data4/shaozhen.liu /data4/shaozhen.liu/datasets /data/shenglin.zeng; do
 echo "CHECK=$r read=$(test -r "$r" && echo yes || echo no) write=$(test -w "$r" && echo yes || echo no) execute=$(test -x "$r" && echo yes || echo no)"
done
find /data4/shaozhen.liu -maxdepth 3 -type d 2>&1 | head -45
find /data/shenglin.zeng -maxdepth 4 -type d \( -iname '*data*' -o -iname '*sdxl*' -o -iname '*laion*' \) 2>/dev/null | head -40
