set -eu
ROOT=/data5/shenglin.zeng
/usr/bin/python3 - <<'PY'
import zipfile
zipfile.ZipFile('/data5/shenglin.zeng/tmp/SparkDiffusion-source.zip').extractall('/data5/shenglin.zeng/src')
PY
cd "$ROOT/src/SparkDiffusion-main"
cat AGENTS.md
printf '\nDISTILL MODEL FILES\n'
find sparkdiffusion -iname '*distill*' -o -iname '*dataset*'
printf '\nDATALOADER\n'
cat sparkdiffusion/configs/defaults/dataloader.py
printf '\nINSTALL STATUS\n'
tail -4 "$ROOT/logs/install_torch.log"
printf '\nMODEL STATUS\n'
tail -12 "$ROOT/logs/download_models.log"
