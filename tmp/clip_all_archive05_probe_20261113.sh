python3 - <<'PY'
import tarfile,collections
p='/data5/shenglin.zeng/checkpoints/a800_archive_20261105/evidence/part_000.tar'
with tarfile.open(p) as t:
 names=t.getnames()
 for n in names:
  if n.endswith(('manifest.json','config.json')):print(n)
 print('PNGS',len([n for n in names if '/images/' in n and n.endswith('.png')]))
 print('\n'.join([n for n in names if '/images/' in n][:5]))
PY
