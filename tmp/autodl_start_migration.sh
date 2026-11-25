set -eu
root=/root/autodl-tmp/sdxl_research
nohup bash "$root/tmp/migrate_system.sh" > "$root/logs/system_migration.log" 2>&1 < /dev/null &
echo MIGRATION_PID=$!
