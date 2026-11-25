cd /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src
grep -r -n -A26 'sftpUnknown\|sftpReply.*=\|sftpDone.*,' putty | head -70
grep -rinE 'sftpRequest|sftpAsk|sftpPassword|sftpHostKey|sftpDone' putty/unix |head -20
grep -n -A40 'verify_ssh_host_key\|verify_host_key' putty/unix/uxcons.c |tail -50

