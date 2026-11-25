date -Is
cat /proc/sys/kernel/yama/ptrace_scope 2>/dev/null || true
command -v py-spy || true
command -v gdb || true
cat /proc/33807/wchan 2>/dev/null || true
