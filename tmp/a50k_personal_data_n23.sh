for r in /data/shenglin.zeng /data5/shenglin.zeng/checkpoints/a800_archive_20261106/datasets; do
 if [ -d "$r" ]; then
  echo "ROOT=$r"
  find "$r" -maxdepth 3 -type d 2>/dev/null | head -60
  find "$r" -maxdepth 8 -type f \( -name prepared.pt -o -name 'vae_latents_*.pt' -o -iname '*20k*.tar*' -o -iname '*50k*.tar*' \) -printf '%p %s bytes\n' 2>/dev/null | head -50
  find "$r" -maxdepth 3 -type l -printf 'LINK %p -> %l\n' 2>/dev/null | head -20
 fi
done
ls -ld /data4 /data4/shaozhen.liu /data4/shenglin.zeng 2>/dev/null || true
find /data5/shenglin.zeng/checkpoints -maxdepth 4 -type d -iname '*dataset*' 2>/dev/null | head -30
