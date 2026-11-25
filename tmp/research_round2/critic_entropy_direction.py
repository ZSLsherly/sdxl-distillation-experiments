"""Tests conditional-tangent entropy identity and hidden-latent weighting bias.
Identity is standard continuity/Stein calculus, not a novelty claim.
"""
from pathlib import Path
import json
import numpy as np
from critic_multimodal import SCENARIOS,at,stats,drift,grid,integrate
ROOT=Path(__file__).resolve().parent

gaussian=[]
rng=np.random.default_rng(20261015)
for tau in [.03,.1,.3,1.,3.]:
    theta=1.;V=theta**2+tau**2
    x=np.linspace(-12*np.sqrt(V),12*np.sqrt(V),20001)
    q=np.exp(-x*x/(2*V))/np.sqrt(2*np.pi*V)
    mz=theta*x/V
    true=-theta/V
    # Wrong same-sample weight W=Z^2: ratio E[W*(-eps/tau)|x]/E[W|x].
    sw=-x/V+2*theta**2*x/(theta**2*x**2+tau**2*V)
    wrong=float(integrate(x,q*mz*sw))
    # Learn m(x)=E[Z|x] by linear regression and use -E m'(x).
    # Compare same N paired samples to linear score DSM then pull back by theta.
    count=500;N=1024
    z=rng.normal(size=(count,N));eps=rng.normal(size=(count,N))
    xx=theta*z+tau*eps
    tangent_slope=np.sum(xx*z,axis=1)/np.sum(xx*xx,axis=1)
    score_slope=np.sum(xx*(-eps/tau),axis=1)/np.sum(xx*xx,axis=1)
    tangent_est=-tangent_slope
    score_est=theta*score_slope
    gaussian.append(dict(noise=tau,true_negentropy_derivative=true,
      conditional_tangent_exact=-theta/V,naive_noise_label_pullback=0.,
      hidden_weight_Z2_limit=wrong,hidden_weight_sign_wrong=bool(wrong*true<0),
      tangent_regression_rmse=float(np.sqrt(np.mean((tangent_est-true)**2))),
      score_DSM_regression_rmse=float(np.sqrt(np.mean((score_est-true)**2))),
      repeats=count,samples_per_repeat=N))

mixture=[]
for name,sc in SCENARIOS.items():
  for tau in [.02,.1,.3,1.]:
    x=grid(sc,tau,16385);mu,sd=at(sc,0);w=np.asarray(sc['w'])
    st=stats(x,mu,sd,w,tau)
    v,vp,_=drift(x,st,sd,np.asarray(sc['dm']),np.asarray(sc['ds']))
    score_form=float(integrate(x,st['q']*st['s']*v))
    divergence_form=float(-integrate(x,st['q']*vp))
    # Finite-difference negentropy under the same parameter path.
    du=1e-5
    mp,sp=at(sc,du);mm,sm=at(sc,-du)
    plus=stats(x,mp,sp,w,tau);minus=stats(x,mm,sm,w,tau)
    fd=float((integrate(x,plus['q']*plus['logq'])-integrate(x,minus['q']*minus['logq']))/(2*du))
    ex=float(integrate(x,st['q']*x));ev=float(integrate(x,st['q']*v))
    B=float(integrate(x,st['q']*(x-ex)*(v-ev))/integrate(x,st['q']*(x-ex)**2))
    mixture.append(dict(scenario=name,noise=tau,score_form=score_form,conditional_divergence_form=divergence_form,
                        finite_difference=fd,affine_tangent_estimate=-B,
                        affine_bias=abs(-B-fd)))
    assert abs(score_form-divergence_form)<1e-8
    assert abs(fd-score_form)<1e-7

out={'status':'CPU scalar/mixture examples. Conditional-tangent estimator is a research lead, not a validated scalable method.',
 'identity':'For fixed parameter direction u, m_u(x)=E[J_theta(z)u | X=x], D_u integral q log q = -E_q div m_u.',
 'caveats':['Do not differentiate through current theta when treating the estimated scalar as a gradient coefficient.',
            'Directions selected on the same fitting/evaluation batch can introduce adaptation bias; cross-fit and freeze directions.',
            'Low-rank parameter updates only optimize that subspace; no equivalence to unrestricted training is claimed.',
            'A tangent field is D-dimensional per direction; this does not automatically reduce network or derivative costs.',
            'Tangent regression can be misspecified, particularly near multimodal posterior overlap; affine approximation is included.'],
 'gaussian':gaussian,'mixture':mixture}
(ROOT/'critic_entropy_direction_results.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({'gaussian':gaussian,'mixture_max_identity_error':max(abs(r['score_form']-r['conditional_divergence_form']) for r in mixture),
                  'mixture_affine_bias_max':max(r['affine_bias'] for r in mixture)},indent=2))
