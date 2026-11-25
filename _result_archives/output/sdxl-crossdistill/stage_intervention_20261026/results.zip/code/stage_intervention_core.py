"""Deterministic VP/DDIM stage replacement, with explicit noisy boundaries."""
def interval_grid(start,end,stride=10):
    assert start>end>=-1 and stride>0
    if end==-1:return sorted(set(list(range(start,-1,-stride))+[0]),reverse=True)+[-1]
    return list(range(start,end,-stride))+[end]

def integrate(x,start,end,predict,jump,stride=10):
    grid=interval_grid(start,end,stride)
    for t,s in zip(grid,grid[1:]):x=jump(x,predict(x,t),t,s)
    return x

def student_rollout(x,knots,predict,jump,start_stage=0):
    states=[x]
    for k in range(start_stage,4):
        x=jump(x,predict(x,knots[k]),knots[k],knots[k+1]);states.append(x)
    return states

def replace_stage(states,stage,knots,student,teacher,jump,stride=10):
    # Copy the SAME on-policy student input. A middle endpoint remains x_s, not x_0.
    x=integrate(states[stage].clone(),knots[stage],knots[stage+1],teacher,jump,stride)
    boundary=x.clone()
    final=student_rollout(x,knots,student,jump,start_stage=stage+1)[-1]
    return final,boundary
