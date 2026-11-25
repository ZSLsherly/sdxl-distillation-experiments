"""Analytic mixture stress test for critic score transport (CPU, no training).

Uses analytic densities, scores, input derivatives, and conditional drifts.
All errors and gradient moments are integrated on a deterministic grid.
No claimed diffusion-video or fixed-2-NFE performance is measured here.
"""
from pathlib import Path
import json, csv, math
import numpy as np

ROOT=Path(__file__).resolve().parent
ROOT.mkdir(parents=True,exist_ok=True)
SQRT2PI=np.sqrt(2*np.pi)
SCENARIOS={
 'separated':dict(mu=[-2.,0.,2.],sd=[.3,.25,.35],w=[.3,.4,.3],dm=[.45,-.35,.25],ds=[.08,-.03,.05]),
 'merging':dict(mu=[-.6,.6],sd=[.18,.18],w=[.5,.5],dm=[.8,-.8],ds=[0.,0.]),
 'rare_mode':dict(mu=[-1.3,1.3],sd=[.2,.35],w=[.85,.15],dm=[-.3,.8],ds=[.03,-.05]),
 'translation':dict(mu=[-1.,1.],sd=[.25,.25],w=[.5,.5],dm=[.8,.8],ds=[0.,0.])
}
NOISES=[.02,.1,.3,1.]
DELTAS=[.01,.025,.05,.1,.2,.4]

def at(sc,u):
    return np.asarray(sc['mu'])+u*np.asarray(sc['dm']), np.asarray(sc['sd'])+u*np.asarray(sc['ds'])

def stats(x,mu,sd,w,tau):
    V=sd**2+tau**2
    centered=x[:,None]-mu[None,:]
    logits=np.log(w)[None,:]-.5*np.log(2*np.pi*V)[None,:]-.5*centered**2/V[None,:]
    maxlog=logits.max(axis=1,keepdims=True)
    raw=np.exp(logits-maxlog)
    r=raw/raw.sum(axis=1,keepdims=True)
    logq=maxlog[:,0]+np.log(raw.sum(axis=1))
    q=np.exp(logq)
    component_s=-centered/V[None,:]
    component_sp=-np.ones_like(centered)/V[None,:]
    s=np.sum(r*component_s,axis=1)
    sp=np.sum(r*(component_s**2+component_sp),axis=1)-s**2
    return dict(q=q,logq=logq,r=r,s=s,sp=sp,ak=component_s,apk=component_sp,V=V,centered=centered)

def drift(x,st,sd,dm,ds):
    beta=dm[None,:]+ds[None,:]*sd[None,:]*st['centered']/st['V'][None,:]
    bp=ds*sd/st['V']
    ar=st['ak']-st['s'][:,None]
    v=np.sum(st['r']*beta,axis=1)
    vp=np.sum(st['r']*(bp[None,:]+ar*beta),axis=1)
    vpp=np.sum(st['r']*(2*ar*bp[None,:]+(ar**2+st['apk']-st['sp'][:,None])*beta),axis=1)
    return v,vp,vpp

def integrate(x,v):
    return np.trapezoid(v,x=x,axis=0)

def rmse(x,q,e): return float(np.sqrt(max(0.,integrate(x,q*e**2))))

def grid(sc,tau,n=8193):
    all_mu=np.concatenate([at(sc,u)[0] for u in [-.4,0,.4,.7]])
    all_sd=np.concatenate([at(sc,u)[1] for u in [-.4,0,.4,.7]])
    pad=10*np.sqrt(all_sd.max()**2+tau**2)
    return np.linspace(all_mu.min()-pad,all_mu.max()+pad,n)

def critic_error(x,kind,eta,score_scale,previous=False):
    if kind=='exact': return np.zeros_like(x),np.zeros_like(x)
    amp=eta*score_scale
    if kind=='constant_bias': return np.full_like(x,amp),np.zeros_like(x)
    # The error is a deterministic, smooth perturbation, not pixelwise iid noise.
    # independent_phase = independent snapshot artifacts; shared = persistent bias.
    parts=kind.split('_')
    omega=float(parts[-1])
    phase=.37+(1.71 if previous and 'independent' in kind else 0.)
    return amp*np.sqrt(2)*np.sin(omega*x+phase), amp*np.sqrt(2)*omega*np.cos(omega*x+phase)

def gradient_metrics(x,new,mu,sd,w,tau,est,teacher,step=.02):
    # J_mu is a component indicator. J_logsd=sd*z under the clean generator.
    # Integrating conditional moments avoids Monte Carlo noise.
    mz=sd[None,:]*new['centered']/new['V'][None,:]
    Jmean=np.concatenate([new['r'],new['r']*sd[None,:]*mz],axis=1)
    true_d=new['s']-teacher['s']
    d=est-teacher['s']
    g=integrate(x,new['q'][:,None]*Jmean*d[:,None])
    gtrue=integrate(x,new['q'][:,None]*Jmean*true_d[:,None])
    ge=g-gtrue
    denom=np.linalg.norm(g)*np.linalg.norm(gtrue)
    cos=float(np.dot(g,gtrue)/denom) if denom>1e-12 else None
    # Exact second moment of per-sample PARAMETER error from the wrong critic.
    Jnorm2=np.sum(new['r']*(1+sd[None,:]**2*(tau**2/new['V'][None,:]+mz**2)),axis=1)
    err=est-new['s']
    second=float(integrate(x,new['q']*err**2*Jnorm2))
    trace_cov=max(0.,second-float(np.dot(ge,ge)))
    kl=float(integrate(x,new['q']*(new['logq']-teacher['logq'])))
    # Same parameter displacement for every method; not a tuned optimizer result.
    gn=np.linalg.norm(g)
    direction=g/(gn+1e-30)
    K=len(mu)
    mu_after=mu-step*direction[:K]
    sd_after=sd*np.exp(-step*direction[K:])
    after=stats(x,mu_after,sd_after,w,tau)
    kl_after=float(integrate(x,after['q']*(after['logq']-teacher['logq'])))
    return dict(gradient_cosine=cos,gradient_bias_norm=float(np.linalg.norm(ge)),
                gradient_true_norm=float(np.linalg.norm(gtrue)),
                gradient_error_rmse_batch8=float(np.sqrt(np.dot(ge,ge)+trace_cov/8)),
                gradient_error_rmse_batch32=float(np.sqrt(np.dot(ge,ge)+trace_cov/32)),
                gradient_error_rmse_batch128=float(np.sqrt(np.dot(ge,ge)+trace_cov/128)),
                kl_before=kl,kl_after_fixed_step=kl_after,kl_improvement=kl-kl_after)

def run_case(name,sc,tau,delta,kind,eta,n=8193):
    x=grid(sc,tau,n)
    w=np.asarray(sc['w']); mu,sd=at(sc,0); mun,sdn=at(sc,delta); mup,sdp=at(sc,-delta)
    old=stats(x,mu,sd,w,tau); new=stats(x,mun,sdn,w,tau); prev=stats(x,mup,sdp,w,tau)
    mt,st=at(sc,.7); teacher=stats(x,mt,st,w,tau)
    v,vp,vpp=drift(x,old,sd,np.asarray(sc['dm']),np.asarray(sc['ds']))
    scale=np.sqrt(integrate(x,old['q']*old['s']**2))
    e,ep=critic_error(x,kind,eta,scale)
    eprev,_=critic_error(x,kind,eta,scale,previous=True)
    shat=old['s']+e; shatp=old['sp']+ep; sprev=prev['s']+eprev
    ex=integrate(x,old['q']*x); ev=integrate(x,old['q']*v)
    B=integrate(x,old['q']*(x-ex)*(v-ev))/integrate(x,old['q']*(x-ex)**2)
    b=ev-B*ex
    affine_v=B*x+b
    fields={
      'stale':shat,
      'ema':.5*shat+.5*sprev,
      'extrapolation':2*shat-sprev,
      'transport_oracle_drift':shat-delta*(vpp+vp*shat+v*shatp),
      'transport_affine_drift':shat-delta*(B*shat+affine_v*shatp),
    }
    rows=[]
    for method,est in fields.items():
        row=dict(scenario=name,noise=tau,delta=delta,error=kind,error_eta=eta,method=method,
                 score_rmse=rmse(x,new['q'],est-new['s']),
                 old_error_rmse=rmse(x,new['q'],e),
                 old_score_scale=float(scale),affine_drift_rmse=rmse(x,old['q'],affine_v-v))
        row.update(gradient_metrics(x,new,mun,sdn,w,tau,est,teacher))
        rows.append(row)
    return rows

def main():
    errors=[('exact',0),('constant_bias',.1)]
    for omega in [1,5,20,80]:
        for correlation in ['shared','independent']:
            errors.append((f'sin_{correlation}_{omega}',.1))
    rows=[]
    # PDE derivative is independently verified against a finite difference in u.
    derivative_checks=[]
    for name,sc in SCENARIOS.items():
        for tau in NOISES:
            x=grid(sc,tau); w=np.asarray(sc['w']); mu,sd=at(sc,0)
            old=stats(x,mu,sd,w,tau)
            v,vp,vpp=drift(x,old,sd,np.asarray(sc['dm']),np.asarray(sc['ds']))
            du=1e-5
            mp,sp=at(sc,du); mm,sm=at(sc,-du)
            numerical=(stats(x,mp,sp,w,tau)['s']-stats(x,mm,sm,w,tau)['s'])/(2*du)
            analytic=-(vpp+vp*old['s']+v*old['sp'])
            error=rmse(x,old['q'],analytic-numerical)
            derivative_checks.append(dict(scenario=name,noise=tau,pde_vs_fd_rmse=error))
            assert error<1e-5,(name,tau,error)
            for delta in DELTAS:
                for kind,eta in errors:
                    rows.extend(run_case(name,sc,tau,delta,kind,eta))
    # Check integration sensitivity in difficult low-noise/high-frequency settings.
    grid_checks=[]
    for name,kind in [('merging','exact'),('translation','sin_shared_80'),('rare_mode','sin_independent_80')]:
        a=run_case(name,SCENARIOS[name],.02,.2,kind,.1 if kind!='exact' else 0,8193)
        b=run_case(name,SCENARIOS[name],.02,.2,kind,.1 if kind!='exact' else 0,16385)
        for ra,rb in zip(a,b):
            change=max(abs(ra[k]-rb[k]) for k in ['score_rmse','gradient_bias_norm','gradient_error_rmse_batch32','kl_improvement'])
            grid_checks.append(dict(scenario=name,error=kind,method=ra['method'],max_abs_change=change))
            assert change<1e-7,(name,kind,ra['method'],change)
    with (ROOT/'critic_multimodal_all.csv').open('w',newline='',encoding='utf-8-sig') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    groups={}
    for row in rows:
        key=(row['scenario'],row['noise'],row['delta'],row['error'],row['error_eta'])
        groups.setdefault(key,{})[row['method']]=row
    summary={}
    for bucket,predicate in [('exact',lambda k:k[3]=='exact'),('corrupt_all',lambda k:k[3]!='exact'),
                             ('high_frequency',lambda k:k[3].endswith('_80')),
                             ('low_noise_exact',lambda k:k[1]<=.1 and k[3]=='exact'),
                             ('small_step_exact',lambda k:k[2]<=.05 and k[3]=='exact')]:
        selected=[g for k,g in groups.items() if predicate(k)]
        entry={'num_cases':len(selected)}
        for method in ['ema','extrapolation','transport_oracle_drift','transport_affine_drift']:
            entry[method]={
                'fraction_better_score_than_stale':float(np.mean([g[method]['score_rmse']<g['stale']['score_rmse'] for g in selected])),
                'median_score_rmse_ratio_to_stale':float(np.median([g[method]['score_rmse']/g['stale']['score_rmse'] for g in selected])),
                'fraction_better_batch32_gradient_error_than_stale':float(np.mean([g[method]['gradient_error_rmse_batch32']<g['stale']['gradient_error_rmse_batch32'] for g in selected])),
                'fraction_negative_kl_improvement':float(np.mean([g[method]['kl_improvement']<0 for g in selected])),
            }
        summary[bucket]=entry
    selected=[]
    for key,g in groups.items():
        name,tau,delta,kind,eta=key
        if (name=='translation' and tau==.02 and delta==.1 and kind in ['exact','sin_shared_80','sin_independent_80']) or \
           (name=='merging' and tau==.02 and delta in [.05,.2,.4] and kind=='exact'):
            selected.extend(g.values())
    output=dict(status='Analytic 1D mixture CPU stress tests; no learned neural critic or real 2-NFE diffusion model.',
      settings=dict(scenarios=SCENARIOS,noises=NOISES,deltas=DELTAS,errors=errors,grid_points=8193,
                    teacher_path_coordinate=.7,parameter_step_norm=.02,
                    budget_note='EMA/extrapolation use an additional historical score snapshot; transport uses exact conditional drift and score input derivative. Costs are NOT matched by this analytic experiment.'),
      derivative_checks=derivative_checks,grid_checks=grid_checks,summary=summary,selected_cases=selected)
    (ROOT/'critic_multimodal_results.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'rows':len(rows),'summary':summary,'max_derivative_error':max(r['pde_vs_fd_rmse'] for r in derivative_checks),
                      'max_grid_change':max(r['max_abs_change'] for r in grid_checks)},indent=2))

if __name__=='__main__': main()
