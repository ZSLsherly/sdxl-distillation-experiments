from pathlib import Path
def replace(path,edits):
    s=path.read_text(encoding='utf-8-sig')
    for old,new,n in edits:
        if s.count(old)!=n:raise ValueError((path,old,s.count(old)))
        s=s.replace(old,new)
    compile(s,str(path),'exec')
    path.write_text(s,encoding='utf8')
def patch(base):
    replace(base/'boundary.py',[
        ('def critic_stage(update, paired=False, seed=20261017):\n    if paired:', 'def critic_stage(update, paired=False, seed=20261017, updates_per_student=2):\n    if paired and updates_per_student == 2:',1)])
    replace(base/'train.py',[
        ('args.objective!="boundary" or args.fake_updates!=2','args.objective!="boundary" or (args.fake_updates!=2 and not (args.both_middle and args.fake_updates==3))',1),
        ('requires boundary objective and exactly two fake updates','requires two boundary updates, or three with both middle DMs',1),
        ('        config[\'fake_stage_policy\']=\'paired_both_middle_then_terminal\'','''        config['fake_stage_policy']='paired_both_middle_then_terminal'
        if args.fake_updates==3:
            config['fake_stage_policy']='separate_2_then_3_then_4'
            config['experiment_modules']['fake']='one independent optimizer update per stage: 2, 3, 4' ''',1),
        ("args.pair_fake_stages,args.seed)","args.pair_fake_stages,args.seed,args.fake_updates)",3),
        ('both = args.both_middle and score_stage','both = args.both_middle and args.fake_updates==2 and score_stage',1),
        ("'fake_supervised_stages':[2,3] if fake_repeat==0 else [4]","'fake_supervised_stages':([2+fake_repeat] if args.fake_updates==3 else ([2,3] if fake_repeat==0 else [4]))",1)])
if __name__=='__main__':
    import sys
    patch(Path(sys.argv[1]))
