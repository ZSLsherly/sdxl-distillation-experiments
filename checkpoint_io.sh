ROOT=/data5/shenglin.zeng
printf 'DISK ACTIVITY\n'
vmstat 1 3
printf '\nWORKER IO\n'
for pid in 2987809 2987810 2987811 2987812; do
 if [ -r "/proc/$pid/io" ]; then printf '%s ' "$pid"; grep -E 'write_bytes|read_bytes' "/proc/$pid/io"; fi
done
printf '\nCHECKPOINT METADATA\n'
find "$ROOT/outputs/sparkdiffusion/CrossDistill_Smoke/wan1.3B_dense_pcm_dmd_smoke/checkpoints" -maxdepth 3 -type f -name .metadata -print
