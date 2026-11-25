r=/data5/shenglin.zeng/tools/filezilla/app
export LD_LIBRARY_PATH="$r/usr/lib/x86_64-linux-gnu"
{ echo '-0-'; echo '-1-'; echo 'open "root@connect.nma1.seetacloud.com" 14892'; sleep 8; } | timeout 12 "$r/usr/bin/fzsftp" | head -35
strings "$r/usr/bin/fzsftp" | grep -E '^reget$|^get$|^open$|^mtime$|^keyfile$|^trust' | head -15

