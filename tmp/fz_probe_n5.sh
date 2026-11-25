r=/data5/shenglin.zeng/tools/filezilla/app
export LD_LIBRARY_PATH="$r/usr/lib/x86_64-linux-gnu"
"$r/usr/bin/filezilla" --version
printf 'help\nquit\n' | timeout 5 "$r/usr/bin/fzsftp" 2>&1 | head -65
