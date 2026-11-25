from pathlib import Path
d=Path('output/sdxl-crossdistill/a50k_train60_20261124/deploy')
p=d/'train.py'
s=p.read_text(encoding='utf-8')
def sub(before,after,n=1):
    global s
    assert s.count(before)==n,before
    s=s.replace(before,after)
sub('    mesh = init_device_mesh', '''    # Half-open ranges in a single shuffled permutation, with exact checkpoint boundaries.
    milestones = {20000,30000,40000,50000}
    assert len(order)==50000 and args.batch==64 and args.fake_updates==2
    batches = []
    start_idx = 0
    for end_idx in sorted(milestones):
        while start_idx < end_idx:
            width = min(args.batch, end_idx-start_idx)
            batches.append((start_idx,width))
            start_idx += width
    consumed = [0] + [start+width for start,width in batches]
    mesh = init_device_mesh''')
sub('positions = plan.positions(update, micro, rank)','positions = [batches[update][0]+j for j in plan.positions(0,micro,rank)]')
sub('positions=plan.positions(update,micro,rank)','positions=[batches[update][0]+j for j in plan.positions(0,micro,rank)]')
sub('        positions = plan.rank_positions(update, rank)','''        # Use the current student batch for both fake fits. Give the two fits separate noise.
        positions = [batches[gcount][0]+j for j in plan.rank_positions(0,rank)]''')
sub('        score_stage = boundary_dm.critic_stage(update,','        positions = [p + (update % args.fake_updates)*len(order) for p in positions]\n        score_stage = boundary_dm.critic_stage(update,')
sub('order[pos % len(order)]','order[pos]',2)
sub('gcount * plan.global_batch','consumed[gcount]',3)
sub('    while gcount < limit or args.matched_wall_seconds is not None:', '''    assert limit == len(batches)
    final_training_state = None
    while gcount < limit:
        width = batches[gcount][1]
        plan = BatchPlan(width,world,min(requested.micro_batch,width//world))''')
sub('            if args.checkpoint_mode=="full" and args.save_every_updates and gcount % args.save_every_updates == 0:\n                snapshot("periodic", "student_update_complete")','''            if consumed[gcount] in milestones:
                final_training_state = str(out/'training_state'/f'samples{consumed[gcount]:05d}_g{gcount:06d}')
                save_training(final_training_state, models, optimizers,
                              dict(gcount=gcount,fcount=fcount,dcount=dcount,student_samples=consumed[gcount]),config)
                print('TRAINING_STATE_SAVED',rank,final_training_state,flush=True)''')
sub('gcount % math.ceil(args.eval_every_samples / plan.global_batch)','gcount % math.ceil(args.eval_every_samples / requested.global_batch)')
a=s.index('    if not args.benchmark and last_saved_gcount != gcount:')
b=s.index('    peak = torch.tensor',a)
s=s[:a]+s[b:]
sub('        status["training_state"] = final_training_state','        status["student_samples"] = consumed[gcount]\n        status["training_state"] = final_training_state')
p.write_text(s,encoding='utf-8')
p=d/'queue50.py';s=p.read_text(encoding='utf-8').split('def checkpoint(')[0]+'''base.update(micro_batch=16,max_student_updates=784)
if shutil.disk_usage(root).free < 255*1024**3:raise RuntimeError("Insufficient space for four resumable states")
raise SystemExit(run("A50K_singlepass",base))
'''
p.write_text(s,encoding='utf-8')
p=Path('tmp/a50k_archive60_n24.py')
s=p.read_text(encoding='utf-8').replace('for g in (320,480,640,800):','for g in (313,470,627,784):').replace("out=root/'outputs'/('A50K_G%d'%g)","out=root/'outputs'/'A50K_singlepass'").replace("dest='/data/shenglin.zeng/a50k_20261123/checkpoints/A50K_G%d'%g","dest='/data/shenglin.zeng/a50k_20261123/checkpoints/singlepass/'+src.name")
p.write_text(s,encoding='utf-8')
