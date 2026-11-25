set -eu
cd /root/autodl-tmp/sdxl_research
cat setup.status
tail -n 5 logs/setup.log
ps -p 1247 -o pid,etime,stat,args || true
ls -lh transfers
