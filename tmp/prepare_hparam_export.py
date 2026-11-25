from pathlib import Path
p=Path('sdxl_export.py').read_text()
p=p.replace('import json,torch,shutil','import json,torch,shutil,argparse')
p=p.replace("root=Path('/data5/shenglin.zeng/sdxl_crossdistill');out=root/'outputs/smoke_1024'", "ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()\nroot=Path('/data5/shenglin.zeng/sdxl_crossdistill');out=root/'outputs'/args.run")
Path('sdxl_hparam_export.py').write_text(p)
for f in ['sdxl_hparam_train.py','sdxl_teacher_stages.py','sdxl_hparam_export.py']:compile(Path(f).read_text(),f,'exec')
