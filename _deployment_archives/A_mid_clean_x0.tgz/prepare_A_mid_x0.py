"""Private copy of archived A. Expose x0 to the score objective; keep DDIM trajectory."""
from pathlib import Path

def patch(name, rules):
    p=Path(name); s=p.read_text()
    for old,new in rules:
        assert s.count(old)==1,old
        s=s.replace(old,new)
    compile(s,name,'exec');p.write_text(s)

patch('boundary.py',[
('''        x = schedule.jump(x, predict(x, t, condition), t, knots[k])
        if k == stage:
            selected = x''','''        eps = predict(x, t, condition)
        if k == stage:
            selected = schedule.clean(x, eps, t)
        x = schedule.jump(x, eps, t, knots[k])'''),
('''            x = schedule.jump(x, predict(x, t, condition), t, knots[k])
        if k in (middle, 4):
            yield k, x''','''            eps = predict(x, t, condition)
            if k in (middle, 4):
                clean = schedule.clean(x, eps, t)
        with torch.no_grad():
            x = schedule.jump(x, eps, t, knots[k])
        if k in (middle, 4):
            yield k, clean''')])
patch('train.py',[
("scheduler,generated.detach(),knots[k],rnd", "scheduler,generated.detach(),-1,rnd"),
("scheduler,generated,knots[k],rnd","scheduler,generated,-1,rnd"),
("s = knots[boundary_dm.critic_stage(fcount,args.pair_fake_stages,args.seed)]","s = -1  # Middle and terminal supervision are both on clean predictions"),
("'boundary_t':knots[score_stage]", "'boundary_t':-1"),
("dict(objective='boundary_v1', margin=", "dict(objective='boundary_v1', middle_target='clean_x0', margin=")])
