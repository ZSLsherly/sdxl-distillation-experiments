set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
test "$(ps -p 2766791 -o ppid= | tr -d ' ')" = 2746444
kill -TERM 2766791
case "$(ps -p 2765790 -o args=)" in *dymd_prior_8_20261101/download_parallel.py*) kill -TERM 2765790;; *) exit 2;; esac
echo STOPPED_ONLY_THIS_EXPERIMENT_SLOW_DOWNLOADS
