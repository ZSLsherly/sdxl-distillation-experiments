date
hostname
df -h -x tmpfs -x devtmpfs -x overlay
for r in /data1/shenglin.zeng /data2/shenglin.zeng /data3/shenglin.zeng /data4/shenglin.zeng /data5/shenglin.zeng /data6/shenglin.zeng; do
 if [ -d "$r" ]; then
  echo "OWN_ROOT=$r"
  find "$r" -maxdepth 7 -type f \( -name prepared.pt -o -name 'vae_latents_*.pt' -o -name '*dataset*manifest*.json' -o -name validation.json \) -printf '%p %s bytes\n' 2>/dev/null | head -100
 fi
done
for r in /data1 /data2 /data3 /data4 /data5 /data6; do
 if [ -d "$r" ]; then echo "TOP_LEVEL=$r"; ls -d "$r"/datasets "$r"/dataset "$r"/public "$r"/share "$r"/shared "$r"/shenglin.zeng 2>/dev/null || true; fi
done
