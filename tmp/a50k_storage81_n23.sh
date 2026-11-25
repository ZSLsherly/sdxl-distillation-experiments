findmnt /data
ls -ld /data /data/shenglin.zeng
echo "DATA_WRITABLE=$(test -w /data && echo yes || echo no)"
if [ -d /data/shenglin.zeng ]; then
 find /data/shenglin.zeng -maxdepth 7 -type f \( -name prepared.pt -o -name 'vae_latents_*.pt' \) -printf '%p %s bytes\n' 2>/dev/null | head -40
 find /data/shenglin.zeng -maxdepth 2 -type d 2>/dev/null | head -30
fi
ls -lh /data5/shenglin.zeng/checkpoints/a800_archive_20261106/datasets/parallel_20k_seed20261018 2>/dev/null || true
