printf 'DISK SPACE\n'
df -hT /home /data[1-7]
printf '\nMOUNT SOURCES\n'
lsblk -o NAME,TYPE,SIZE,ROTA,FSTYPE,MOUNTPOINT
printf '\nUSER DIRECTORY DISTRIBUTION\n'
for d in /data[1-7]; do
 printf '\n%s\n' "$d"
 ls -ld "$d"
 find "$d" -mindepth 1 -maxdepth 1 -type d ! -name lost+found -printf '%f\t%u\n' 2>/dev/null | sort
done
