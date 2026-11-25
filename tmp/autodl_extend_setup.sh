set -eu
cd /root/autodl-tmp/sdxl_research
if test "$(cat setup.status)" = waiting_for_assets && test "$(tr '\0' ' ' < /proc/1112/cmdline)" = 'bash setup.sh '; then
 kill -TERM 1112
 while kill -0 1112 2>/dev/null; do sleep 1; done
 mv setup_next.sh setup.sh
 nohup bash setup.sh >> logs/setup.log 2>&1 < /dev/null &
 echo SETUP_PID=$!
fi
