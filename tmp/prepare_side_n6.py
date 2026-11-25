from pathlib import Path
p=Path('tmp')
s=(p/'dispatch_five_n5.py').read_text()
s=s.replace("token='five_checkpoints_n5'","LAB='/data5/shenglin.zeng/checkpoints/a800_archive_20261106'\ntoken='side_archive_n6'")
s=s.replace("private=LAB+'/.pull5'","private=LAB+'/.pull6'")
s=s.replace('build_five_n5','build_side_n6').replace('read_backup_n5','read_side_n6')
(p/'dispatch_side_n6.py').write_text(s)
(p/'read_side_n6.py').write_text((p/'read_backup_n5.py').read_text())
