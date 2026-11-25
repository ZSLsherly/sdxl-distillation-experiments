from backup_a800_nov5 import lab,run
with lab() as d:
 print(run(d,"r=/data5/shenglin.zeng/tools/filezilla/app; export LD_LIBRARY_PATH=$r/usr/lib/x86_64-linux-gnu; $r/usr/bin/filezilla --version 2>&1 || true; printf 'help\\nquit\\n' | timeout 5 $r/usr/bin/fzsftp 2>&1 || true"))
 print(run(d,"cd /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src; grep -nE 'sftpRequest|sftpReply|fznotify|sftp_cmd_get|cmd_get|verify_ssh|askappend|Password' putty/fzsftp.c putty/fzputtygen.c putty/unix/uxcons.c putty/fznotify.h 2>/dev/null | head -70; cat putty/fznotify.h 2>/dev/null || true"))
