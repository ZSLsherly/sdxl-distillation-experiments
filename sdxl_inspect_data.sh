source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
tail -n 8 "$root/logs/download.log"
tail -n 8 "$root/logs/data_download.log"
python - <<'PY'
import requests
for path in ['main/data/create_lmdb_iterative.py','main/data/sdxl_dataset.py','main/sdxl/sdxl_guidance.py']:
 r=requests.get('https://raw.githubusercontent.com/tianweiy/DMD2/main/'+path,timeout=20)
 print('SOURCE',path,r.status_code)
 if r.status_code==200:print(r.text[:18000])
PY
