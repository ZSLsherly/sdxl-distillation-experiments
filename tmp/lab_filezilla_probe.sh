set -u
id -u
uname -m
cat /etc/os-release | head -5
command -v filezilla fzcli xvfb-run Xvfb apt-get sshpass lftp sftp || true
echo "display=${DISPLAY:-none}"
sudo -n true 2>/dev/null && echo passwordless_sudo || true
du -sh /data5/shenglin.zeng/checkpoints/a800_archive_20261105
timeout 6 bash -c 'head -c 50 </dev/tcp/connect.nma1.seetacloud.com/14892' || true
apt-cache policy filezilla | head -6
