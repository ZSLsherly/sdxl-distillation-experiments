set -eu
# Stop only the exact CPU validator launched in this task, not any GPU task.
if [ -r /proc/2476643/cmdline ]; then
  cmd=$(tr '\0' ' ' < /proc/2476643/cmdline)
  case "$cmd" in
    '/data1/shenglin.zeng/dmd2_official_recipe_20261121/env/bin/python test_recipe.py ')
      kill -TERM 2476643
      echo STOPPED_OWN_CPU_VALIDATOR_2476643;;
    *) echo PID_NOT_MATCHED_NO_ACTION;;
  esac
fi
