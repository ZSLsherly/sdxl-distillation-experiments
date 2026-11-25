R=/data1/shenglin.zeng
find "$R" -maxdepth 5 -type f \( -name '*manifest*.json' -o -name 'revision.txt' -o -name '*real*.npz' -o -name 'evaluation_core.py' -o -name 'features*.npz' \) 2>/dev/null | grep -E 'coco|clip|semantic|eval' | head -n 90
find "$R" -maxdepth 4 -type d \( -iname '*clip*' -o -iname '*coco*' \) | head -n 45
df -h /data1
