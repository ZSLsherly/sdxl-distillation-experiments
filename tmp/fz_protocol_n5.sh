cd /data5/shenglin.zeng/tools/filezilla/src/filezilla-3.46.3/src
sed -n '125,205p' putty/fzsftp.c
grep -rinE 'sftpRequest|sftpAsk|sftpDone|host key|sftpIo' engine/sftp | head -40
grep -r -n -A25 'enum sftp' putty | head -50

