from backup_a800_nov5 import lab,run
with lab() as d:
 print(run(d,"cd /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src; grep -rn -A40 'enum sftp' putty | head -65; sed -n '149,245p' putty/unix/uxcons.c; sed -n '475,545p' putty/unix/uxcons.c; grep -nE 'sftp_cmd_get|cmd_get|\"get\"|\"reget\"' putty/psftp.c"))
