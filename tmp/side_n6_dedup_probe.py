from backup_a800_nov5 import cloud,lab,run
with lab() as d:
 print(run(d,'python3 /data5/shenglin.zeng/checkpoints/a800_archive_20261106/.pull6/stop_own.py'))
 print(run(d,"find /data5/shenglin.zeng -maxdepth 4 -type f \\( -name '*.tar' -o -name '*.tar.gz' -o -name '*manifest*.json' -o -name '*SHA*' -o -name '*sha*' \\) -printf '%p %s\\n'; ls /data5/shenglin.zeng/tools/filezilla; command -v Xvfb xvfb-run || true; /data5/shenglin.zeng/tools/filezilla/app/usr/bin/filezilla --version 2>&1 || true"))
 print(run(d,"sed -n '125,205p' /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src/putty/fzsftp.c"))
