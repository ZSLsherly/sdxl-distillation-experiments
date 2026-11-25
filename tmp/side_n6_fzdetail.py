from backup_a800_nov5 import lab,run
with lab() as d:
 print(run(d,"sed -n '1,100p' /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src/putty/fzprintf.c; sed -n '130,205p' /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src/putty/fzprintf.c; sed -n '800,840p' /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src/putty/psftp.c"))
