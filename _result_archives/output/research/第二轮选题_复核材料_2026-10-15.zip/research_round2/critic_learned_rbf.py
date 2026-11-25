"""Finite-capacity RBF fake critics; complements oracle-perturbation tests.
Not pretrained diffusion critics and not an equal-wall-time training benchmark.
"""
from pathlib import Path
import json,csv
import numpy as np
from critic_multimodal import SCENARIOS,at,stats,drift,grid,integrate,rmse,gradient_metrics
ROOT=Path(__file__).resolve().parent

def features(x,centers,width):
    dx=x[:,None]-centers[None,:]
    r=np.exp(-.5*(dx/width)**2)
    return np.concatenate([np.ones((len(x),1)),r],axis=1),np.concatenate([np.zeros((len(x),1)),-dx/width**2*r],axis=1)

def critic_fit(sc,u,tau,centers,width,seed,n=4096,population=False):
    mu,sd=at(sc,u);w=np.asarray(sc['w'])
    avg=np.sum(w*mu); variance=np.sum(w*(sd**2+tau**2+(mu-avg)**2))
    if population:
        x=grid(sc,tau,4097);st=stats(x,mu,sd,w,tau)
        y=st['s']; weights=st['q']*(x[1]-x[0]);weights[[0,-1]]*=.5
    else:
        rng=np.random.default_rng(seed)
        k=rng.choice(len(w),n,p=w); z=rng.normal(size=n);e=rng.normal(size=n)
        x=mu[k]+sd[k]*z+tau*e;y=-e/tau
        weights=np.full(n,1/n)
    F,_=features(x,centers,width)
    base=-(x-avg)/variance
    gram=F.T@(weights[:,None]*F)
    rhs=F.T@(weights*(y-base))
    coefs=np.linalg.solve(gram+1e-3*np.eye(F.shape[1]),rhs)
    def evaluate(xx):
        FF,FD=features(xx,centers,width)
        return -(xx-avg)/variance+FF@coefs,-np.ones_like(xx)/variance+FD@coefs
    return evaluate

rows=[]
for name in ['merging','rare_mode','translation']:
  sc=SCENARIOS[name];w=np.asarray(sc['w'])
  centers=np.linspace(min(sc['mu'])-1.2,max(sc['mu'])+1.2,32)
  for tau in [.02,.1,.3,1.]:
    x=grid(sc,tau,4097)
    mu,sd=at(sc,0);old=stats(x,mu,sd,w,tau)
    v,vp,vpp=drift(x,old,sd,np.asarray(sc['dm']),np.asarray(sc['ds']))
    for delta in [.05,.2]:
      mn,sn=at(sc,delta);new=stats(x,mn,sn,w,tau)
      mt,st=at(sc,.7);teacher=stats(x,mt,st,w,tau)
      for width in [.2,.5]:
        for fit_mode in ['population_ridge','sample_DSM']:
          for seed in ([0] if fit_mode=='population_ridge' else [0,1,2,3,4]):
            population=fit_mode=='population_ridge'
            old_fn=critic_fit(sc,0,tau,centers,width,seed,population=population)
            prev_fn=critic_fit(sc,-delta,tau,centers,width,seed,population=population)
            fresh_fn=critic_fit(sc,delta,tau,centers,width,seed+100,population=population)
            so,spo=old_fn(x);sprev,_=prev_fn(x);fresh,_=fresh_fn(x)
            estimates={'stale':so,'ema':.5*(so+sprev),'extrapolation':2*so-sprev,
              'transport_oracle_drift':so-delta*(vpp+vp*so+v*spo),
              'fresh_current_DSM_fit':fresh}
            # Exact density pullback for translation: structural control avoiding s'.
            if name=='translation':
                shifted,_=old_fn(x-delta*np.asarray(sc['dm'])[0])
                estimates['exact_translation_pullback']=shifted
            for method,est in estimates.items():
                r=dict(scenario=name,noise=tau,delta=delta,width=width,fit_mode=fit_mode,seed=seed,method=method,
                       score_rmse=rmse(x,new['q'],est-new['s']),old_fit_rmse=rmse(x,old['q'],so-old['s']))
                r.update(gradient_metrics(x,new,mn,sn,w,tau,est,teacher))
                rows.append(r)
with (ROOT/'critic_learned_rbf_all.csv').open('w',encoding='utf-8-sig',newline='') as fh:
    wr=csv.DictWriter(fh,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
summary={}
for fit_mode in ['population_ridge','sample_DSM']:
    groups={}
    for r in rows:
        if r['fit_mode']==fit_mode:
            key=(r['scenario'],r['noise'],r['delta'],r['width'],r['seed'])
            groups.setdefault(key,{})[r['method']]=r
    summary[fit_mode]={'num_cases':len(groups)}
    for method in ['ema','extrapolation','transport_oracle_drift','fresh_current_DSM_fit']:
        gs=list(groups.values())
        summary[fit_mode][method]=dict(
          score_win_fraction_vs_stale=float(np.mean([g[method]['score_rmse']<g['stale']['score_rmse'] for g in gs])),
          median_score_ratio_vs_stale=float(np.median([g[method]['score_rmse']/g['stale']['score_rmse'] for g in gs])),
          gradient32_win_fraction_vs_stale=float(np.mean([g[method]['gradient_error_rmse_batch32']<g['stale']['gradient_error_rmse_batch32'] for g in gs])),
          median_gradient32_ratio_vs_stale=float(np.median([g[method]['gradient_error_rmse_batch32']/g['stale']['gradient_error_rmse_batch32'] for g in gs])),
          count_negative_kl_improvement=int(sum(g[method]['kl_improvement']<0 for g in gs)))
metadata={'status':'Learned 1D RBF ridge critics, CPU. No neural diffusion/video training.',
 'notes':['32 Gaussian RBFs plus intercept and fixed Gaussian tail. Width 0.2/0.5; ridge 0.001.',
          'Population-ridge isolates finite capacity; sample-DSM uses 4096 fresh Gaussian corruption pairs.',
          'Old and previous snapshots share latent seeds (favorable correlated errors for extrapolation).',
          'Fresh-current fit is full closed-form refitting on current samples, not one optimizer step.',
          'The synthetic DSM fit is from scratch and lacks pretrained diffusion priors; do not extrapolate failure rates to DMD2.',
          'Oracle conditional drift is still given to transport, not learned; this favors transport.'],
 'summary':summary,
 'selected':[r for r in rows if r['scenario']=='translation' and r['noise']==.02 and r['delta']==.2 and r['width']==.2 and r['seed']==0]}
(ROOT/'critic_learned_rbf_results.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
