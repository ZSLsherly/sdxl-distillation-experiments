set -eu
base=/data5/shenglin.zeng/checkpoints/a800_abc_g160_20261102
find "$base" -maxdepth 2 \( -name '*.pt' -o -name '*.partial' -o -name EXPERIMENTS.txt -o -name VERIFIED.txt \) -printf '%P %s\n'
pgrep -af '^ssh .*abc_g160_transfer_20261102' || true
if test -f "$base/.abc_g160_transfer_20261102/direct.log"; then tail -5 "$base/.abc_g160_transfer_20261102/direct.log"; fi
find "$base/.abc_g160_transfer_20261102" -maxdepth 2 -type f \( -name "*.chunk" -o -name "*.part" \) -printf "%s\n" | awk '{s+=$1;n++} END {printf "chunk files=%s bytes=%.0f\n",n,s}' 
