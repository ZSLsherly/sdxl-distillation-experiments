set -eu
cd /root/autodl-tmp/sdxl_research
nohup bash setup.sh > logs/setup.log 2>&1 < /dev/null &
echo SETUP_PID=$!
ls -lh transfers
