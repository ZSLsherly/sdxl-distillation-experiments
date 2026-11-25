from backup_a800_nov5 import lab,run
with lab() as d:
 print(run(d,"cd /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src; grep -rn -A35 'sftpUnknown\|sftpReply =\|sftpAskHostkey =' putty | head -90; sed -n '630,660p' putty/psftp.c; sed -n '835,860p' putty/psftp.c; grep -rn 'sftpAskHostkey\|sftpEncryption' putty/*.h; sed -n '1,110p' putty/fzsftp.c"))
