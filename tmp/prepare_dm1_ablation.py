"""Prepare an isolated first-step boundary-DM experiment from the deployed three-fake trainer."""
from pathlib import Path
import json
import os
import shutil
import sys

name = 'c_mix1_first_dm_fake3_g160'
previous = 'c_mix1_dm234_fake3_20261103'

def patch(path, replacements):
    source = path.read_text()
    for old, new, count in replacements:
        if source.count(old) != count:
            raise ValueError(f'{path}: expected {count} copies of {old!r}')
        source = source.replace(old, new)
    compile(source, str(path), 'exec')
    path.write_text(source)

def prepare(code):
    # This directory serves this experiment only; preserve old fake label IDs 0,1,2.
    patch(code/'boundary.py', [
        ('for k in (2, 3, 4)}','for k in (1, 2, 3, 4)}',1),
        ('    if paired and updates_per_student == 2:', '''    if paired and updates_per_student == 3:
        cycle, phase = divmod(update, 3)
        return (1, middle_stage(cycle, seed), 4)[phase]
    if paired and updates_per_student == 2:''',1),
        ('((2,), (3,), (2,3))','((1,2), (1,3), (2,), (3,), (2,3))',1),
        ('Embedding(3, width)','Embedding(4, width)',1),
        ('num_class_embeds=3','num_class_embeds=4',1),
        ('    if stage not in (2, 3, 4):','    if stage == 1:\n        return torch.full((len(sample),), 3, device=sample.device, dtype=torch.long)\n    if stage not in (2, 3, 4):',1)])
    patch(code/'train.py', [
        ('    if args.both_middle and (not args.pair_fake_stages', '''    if not (args.objective == "boundary" and args.pair_fake_stages and args.fake_updates == 3
            and not args.both_middle and args.pcm_weight == 0 and args.variant == "dmd_only"
            and not args.calibrate_only and args.boundary_noise_max == 979):
        parser.error("This private entry point requires first + random middle + terminal DM")
    if args.both_middle and (not args.pair_fake_stages''',1),
        ('(args.fake_updates!=2 and not (args.both_middle and args.fake_updates==3))','args.fake_updates!=3',1),
        ('    if args.flash_terminal_gan:\n        config', '''    config['fake_stage_policy']='first_then_selected_middle_then_terminal'
    config['pcm_t_range']=None
    config['turning_sampling']['scope']='student and fake boundary DM'
    config['experiment_modules'].update(student='DM1 + selected DM2/3 + DM4',fake='one update per supervised stage',pcm=False)
    if args.flash_terminal_gan:
        config''',1),
        ('knots, (2,3) if args.both_middle else stage):','knots, (1,stage)):',1),
        ('middle_value = terminal_value = torch.zeros','first_value = middle_value = terminal_value = torch.zeros',1),
        ('                        else:\n                            middle_value','                        elif k == 1:\n                            first_value = dmd.detach()\n                        else:\n                            middle_value',1),
        ('torch.stack([middle_value,terminal_value])','torch.stack([first_value,middle_value,terminal_value])',1),
        ('["middle_dm_loss","terminal_dm_loss"]','["first_dm_loss","middle_dm_loss","terminal_dm_loss"]',1),
        ('''            with measure("G_PCM_backward"):
                pcm = pcm_loss(real,rnd,condition)
                (args.pcm_weight * pcm / plan.accumulation).backward()
                stats.append(torch.stack([dmd_value, gan_value, pcm.detach()]))
                del pcm, real, rnd, condition, dmd_value, gan_value''','''            stats.append(torch.stack([dmd_value, gan_value, torch.zeros_like(dmd_value)]))
            del real, rnd, condition, dmd_value, gan_value''',1),
        ("[2,3,4] if args.both_middle else [stage,4]","[1,stage,4]",1),
        ("**({'fake_supervised_stages':([2+fake_repeat] if args.fake_updates==3 else ([2,3] if fake_repeat==0 else [4]))} if args.both_middle else {}),","'fake_supervised_stages': [primary],",1)])
    patch(code/'run_quota_bc.py', [("==[2,3,4]","==[1,g['stage'],4]",1)])
    patch(code/'run_dm234_fake3.py',[
        ("out=root/'outputs/c_mix1_dm234_fake3_20261103'","out=root/'outputs/c_mix1_first_dm_fake3_g160'",1),
        ("    while True:",'''    upstream=root/'outputs/c_mix1_dm234_fake3_20261103/complete.json'
    print('WAITING_FOR',upstream,flush=True)
    while not upstream.is_file():
        if (out/'STOP_REQUEST').exists(): return
        time.sleep(30)
    done=json.loads(upstream.read_text())
    assert (done['G'],done['fake'])==(160,480)
    while True:''',1),
        ("r['supervised_stages']==[2,3,4]","r['supervised_stages']==[1,r['stage'],4] and r['pcm_loss']==0 and math.isfinite(r['first_dm_loss'])",1),
        ("[[2],[3],[4]]*160","[[k] for g in gs for k in (1,g['stage'],4)]",1)])

if __name__=='__main__':
    root=Path(os.environ['SDXL_ROOT'])
    original=root/'tmp'/previous/'code'
    code=root/'tmp'/name/'code'
    out=root/'outputs'/name
    if out.exists():
        raise SystemExit(f'Queue directory already exists: {out}')
    code.mkdir(parents=True,exist_ok=True)
    for f in original.glob('*.py'):
        shutil.copy2(f,code/f.name)
    prepare(code)
    cfg=json.loads((root/'outputs'/previous/'effective_config.json').read_text())
    cfg.update(both_middle=False,pcm_weight=0)
    out.mkdir()
    (out/'effective_config.json').write_text(json.dumps(cfg,indent=2))

