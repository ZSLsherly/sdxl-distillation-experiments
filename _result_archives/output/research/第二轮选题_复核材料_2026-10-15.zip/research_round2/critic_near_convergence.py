"""Near-target stress test: fixed learning rate, exact population gradient.
All score perturbations are controlled, not neural training results.
"""
from pathlib import Path
import json,csv
import numpy as np
from critic_multimodal import SCENARIOS,at,stats,drift,grid,integrate,critic_error
ROOT=Path(__file__).resolve().parent
rows=[]
for name,sc in SCENARIOS.items():
 for tau in [.02,.1,.3,1.]:
  x=grid(sc,tau);w=np.asarray(sc['w']);mu,sd=at(sc,0)
  old=stats(x,mu,sd,w,tau)
  v,vp,vpp=drift(x,old,sd,np.asarray(sc['dm']),np.asarray(sc['ds']))
  scale=np.sqrt(integrate(x,old['q']*old['s']**2))
  for delta in [.05,.2]:
   mn,sn=at(sc,delta);new=stats(x,mn,sn,w,tau)
   mp,sp=at(sc,-delta);previous=stats(x,mp,sp,w,tau)
   J=np.concatenate([new['r'],new['r']*sn[None,:]**2*new['centered']/new['V'][None,:]],axis=1)
   for gap in [0.,.01]:
    mt,st=at(sc,delta+gap);target=stats(x,mt,st,w,tau)
    kl0=float(integrate(x,new['q']*(new['logq']-target['logq'])))
    gtrue=integrate(x,new['q'][:,None]*J*(new['s']-target['s'])[:,None])
    for error in ['exact','constant_bias','sin_shared_5','sin_shared_80']:
     e,ep=critic_error(x,error,.1 if error!='exact' else 0.,scale)
     shat=old['s']+e;shatprime=old['sp']+ep;prev=previous['s']+e
     fields={'stale':shat,'ema':.5*(shat+prev),'extrapolation':2*shat-prev,
             'transport_oracle_drift':shat-delta*(vpp+vp*shat+v*shatprime),
             'current_true_score':new['s']}
     for method,sest in fields.items():
      g=integrate(x,new['q'][:,None]*J*(sest-target['s'])[:,None])
      step=.01*g
      # Shared clipping threshold, report whether it was activated.
      norm=float(np.linalg.norm(step));clipped=norm>.1
      if clipped:step*=.1/norm
      K=len(mn);after=stats(x,mn-step[:K],sn*np.exp(-step[K:]),w,tau)
      kl1=float(integrate(x,after['q']*(after['logq']-target['logq'])))
      row=dict(scenario=name,noise=tau,delta=delta,target_gap=gap,error=error,method=method,
               kl_before=kl0,kl_after=kl1,kl_change=kl1-kl0,
               gradient_bias_norm=float(np.linalg.norm(g-gtrue)),gradient_norm=float(np.linalg.norm(g)),
               true_gradient_norm=float(np.linalg.norm(gtrue)),clipped=clipped)
      rows.append(row)
with (ROOT/'critic_near_convergence_all.csv').open('w',encoding='utf-8-sig',newline='') as fh:
 wr=csv.DictWriter(fh,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
summary={}
for gap in [0.,.01]:
 for error in ['exact','constant_bias','sin_shared_5','sin_shared_80']:
  key=f'gap={gap},error={error}'
  summary[key]={}
  for method in ['stale','ema','extrapolation','transport_oracle_drift','current_true_score']:
   rs=[r for r in rows if r['target_gap']==gap and r['error']==error and r['method']==method]
   summary[key][method]=dict(cases=len(rs),kl_increase_count=sum(r['kl_change']>1e-10 for r in rs),
                             median_kl_change=float(np.median([r['kl_change'] for r in rs])),
                             median_gradient_bias=float(np.median([r['gradient_bias_norm'] for r in rs])))
out=dict(status='Analytic near-target stress test; exact gradients via quadrature, fixed LR 0.01, parameter step cap 0.1.',
         notes=['Target gap 0 means current student exactly equals teacher; any KL rise is a spurious update.',
                'Target gap 0.01 means teacher lies slightly further along the specified parameter path.',
                'This test excludes minibatch noise; high-frequency score error can cancel in population gradients.',
                'The chosen cases are a transparent fixed grid, not a random sample of real diffusion training.'],
         summary=summary,
         selected=[r for r in rows if r['scenario']=='merging' and r['noise']==.02 and r['delta']==.2 and r['target_gap']==.01])
(ROOT/'critic_near_convergence_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
