"""Analytic Gaussian checks for a proposed influence-weighted DMD critic.

This script trains no neural network. It solves scalar population least-squares
problems by Gauss-Hermite quadrature, with an analytic true student score.
"""
from pathlib import Path
import json
import numpy as np
from numpy.polynomial.hermite import hermgauss
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent

def grid(order=256):
    nodes, weights = hermgauss(order)
    # x=z+epsilon ~ N(0,2). Hermite weights integrate exp(-u^2).
    return 2*nodes, weights/np.sqrt(np.pi)

def score_true(x):
    return -x/2

def conditional_metric(x):
    # J=dG_theta(z)/dtheta=z at theta=1; z|x ~ N(x/2,1/2).
    return .5+x*x/4

def score_naive_weighted(x):
    # argmin_s E[z^2(s(x)+epsilon)^2|x].
    return x*(2-x*x)/(2*(2+x*x))

FEATURES = {
    'linear (representable)': lambda x:x,
    'tanh': np.tanh,
    'rational x/(1+x^2)': lambda x:x/(1+x*x),
    'cubic x+0.25x^3': lambda x:x+.25*x**3,
    'sine': np.sin,
}

def solve(order=256):
    x, prob = grid(order)
    s = score_true(x)
    K = conditional_metric(x)
    sn = score_naive_weighted(x)
    # Fixed in advance, including an exactly representable sanity case and
    # three distinct misspecifications. The cubic and sine are adverse cases.
    methods = {
        'uniform_DSM': (np.ones_like(x), s),
        'oracle_query_metric': (K, s),
        'bounded_query_metric': (.5+1.5*(K-.5)/(1.5+K-.5), s),
        'naive_clean_J2': (K, sn),
    }
    results = []
    for feature_name, feature in FEATURES.items():
        f = feature(x)
        for method, (weight, target) in methods.items():
            a = np.sum(prob*weight*f*target)/np.sum(prob*weight*f*f)
            error = a*f-s
            mean_error = .5*np.sum(prob*x*error)
            results.append({
                'feature':feature_name, 'method':method, 'coefficient':float(a),
                'ambient_score_MSE':float(np.sum(prob*error**2)),
                'per_sample_parameter_gradient_MSE':float(np.sum(prob*K*error**2)),
                'mean_student_gradient_error':float(mean_error),
                'mean_student_gradient_bias_squared':float(mean_error**2),
                # Teacher clean variance 0.25, same corruption variance 1.
                # E[z(s_q(x)-s_T(x))]=(-1/2+1/1.25)E[zx]=0.3.
                'ideal_DMD_gradient_for_teacher_clean_var_0_25':.3,
                'estimated_DMD_gradient':float(.3+mean_error),
            })
    return results

def main():
    results = solve(256)
    low_results = solve(192)
    max_quadrature_delta = max(abs(a['coefficient']-b['coefficient'])
                               for a,b in zip(results,low_results))
    x0 = 1.0
    # Direct conditional quadrature independently checks the biased optimum.
    n,w = hermgauss(128)
    z = x0/2+n  # conditional variance 1/2
    w /= np.sqrt(np.pi)
    numerical_naive = np.sum(w*z*z*(z-x0))/np.sum(w*z*z)
    assert np.isclose(numerical_naive, 1/6, atol=1e-12)
    assert np.isclose(score_true(x0), -.5)
    cubic = {r['method']:r for r in results if r['feature']=='cubic x+0.25x^3'}
    assert np.isclose(cubic['uniform_DSM']['coefficient'], -5/31, atol=1e-12)
    assert np.isclose(cubic['oracle_query_metric']['coefficient'], -1/8, atol=1e-12)
    assert cubic['oracle_query_metric']['per_sample_parameter_gradient_MSE'] < cubic['uniform_DSM']['per_sample_parameter_gradient_MSE']
    assert cubic['oracle_query_metric']['mean_student_gradient_bias_squared'] > cubic['uniform_DSM']['mean_student_gradient_bias_squared']
    payload = {
        'experiment_type':'analytic Gaussian population least squares; no neural network training',
        'setup':{'generator':'G_theta(z)=theta*z at theta=1','z':'N(0,1)',
                 'corruption':'x=z+epsilon, epsilon~N(0,1)',
                 'true_score':'-x/2','DSM_target':'-epsilon',
                 'conditional_metric':'K(x)=E[z^2|x]=1/2+x^2/4',
                 'naive_optimum':'x*(2-x^2)/(2*(2+x^2))',
                 'critic_family':'s_a(x)=a*f(x); one scalar coefficient per family',
                 'bounded_metric':'0.5+1.5*(K-0.5)/(1.5+K-0.5), positive query-only approximation'},
        'pointwise_counterexample':{'x':x0,'true_score':-.5,
            'naive_clean_J2_optimum_formula':float(score_naive_weighted(x0)),
            'naive_clean_J2_optimum_conditional_quadrature':float(numerical_naive)},
        'quadrature':{'order_main':256,'order_check':192,
                      'max_coefficient_difference':float(max_quadrature_delta)},
        'results':results,
        'interpretation':[
            'Query-only positive weights preserve the population score minimizer in an unrestricted family.',
            'Clean-sample Jacobian weights generally change that minimizer; stop-gradient does not remove statistical bias.',
            'An oracle conditional metric minimizes the per-sample gradient second-moment risk within this family by construction.',
            'That risk upper-bounds squared mean gradient bias, but minimizing the bound can increase the actual mean bias.',
            'The cubic family is an explicit adverse case: risk falls while mean bias squared increases by 3.75390625x.',
            'These are mathematical checks, not evidence of improved image or video generation.',
        ]}
    (OUT/'influence_analytic_results.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')

    im=Image.new('RGB',(1840,820),'white')
    draw=ImageDraw.Draw(im)
    font_path='C:/Users/sherl/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/share/fonts/DejaVuSans.ttf'
    fonts={s:ImageFont.truetype(font_path,s) for s in (20,22,24,26,28,40)}
    teal='#087E83'; orange='#D66A35'; ink='#172B42'; muted='#586B7D'; grid_color='#D5DEE5'
    def text(x,y,value,size=24,color=ink,anchor=None):
        draw.text((x,y),value,font=fonts[size],fill=color,anchor=anchor)
    text(920,20,'Influence-weighted critic: exact Gaussian checks',40,anchor='mt')
    text(75,100,'A. Naive weighting changes the score target',26)
    text(990,100,'B. Lower risk can still mean larger mean bias',26)
    left,right,top,bottom=95,855,205,620
    def xy(x,y):return (left+(x+3.2)/6.4*(right-left),bottom-(y+1.8)/3.6*(bottom-top))
    draw.line((left,top,left,bottom,right,bottom),fill=muted,width=2)
    for xv in (-3,-2,-1,0,1,2,3):
        xp,yp=xy(xv,-1.8)
        draw.line((xp,bottom,xp,bottom+6),fill=muted,width=2)
        text(xp,bottom+12,str(xv),22,anchor='mt')
    for yv in (-1.5,-1,-.5,0,.5,1,1.5):
        xp,yp=xy(-3.2,yv)
        draw.line((left,yp,right,yp),fill=grid_color,width=1)
        text(left-10,yp,f'{yv:g}',20,anchor='rm')
    x=np.linspace(-3.2,3.2,500)
    draw.line([xy(float(xx),float(yy)) for xx,yy in zip(x,score_true(x))],fill=teal,width=5)
    draw.line([xy(float(xx),float(yy)) for xx,yy in zip(x,score_naive_weighted(x))],fill=orange,width=5)
    for yy,col in [(-.5,teal),(1/6,orange)]:
        xp,yp=xy(1,yy); draw.ellipse((xp-6,yp-6,xp+6,yp+6),fill=col)
    text(120,150,'True score / query-only SPD',22,teal)
    text(510,150,'Naive clean-J weight',22,orange)
    text(150,225,'At x=1: true -0.5; naive +1/6',22,muted)
    text((left+right)/2,670,'Noisy state x',24,anchor='mt')
    text(40,175,'Score',22)
    labels=['tanh','cubic +','sine']
    names=['tanh','cubic x+0.25x^3','sine']
    risk_ratios=[]; bias_ratios=[]
    for name in names:
        rr={r['method']:r for r in results if r['feature']==name}
        risk_ratios.append(rr['oracle_query_metric']['per_sample_parameter_gradient_MSE']/rr['uniform_DSM']['per_sample_parameter_gradient_MSE'])
        bias_ratios.append(rr['oracle_query_metric']['mean_student_gradient_bias_squared']/rr['uniform_DSM']['mean_student_gradient_bias_squared'])
    left,right,top,bottom=1040,1775,205,620
    def ybar(v):return bottom-v/4.35*(bottom-top)
    draw.line((left,top,left,bottom,right,bottom),fill=muted,width=2)
    for yv in range(5):
        yp=ybar(yv)
        draw.line((left,yp,right,yp),fill=grid_color,width=1)
        text(left-12,yp,str(yv),22,anchor='rm')
    for xx in range(left,right,20):
        draw.line((xx,ybar(1),min(xx+10,right),ybar(1)),fill=muted,width=2)
    text(1030,150,'Per-sample gradient MSE',22,teal)
    text(1415,150,'Squared mean bias',22,orange)
    for i,(r,b) in enumerate(zip(risk_ratios,bias_ratios)):
        mid=left+(i+.5)*(right-left)/3
        draw.rectangle((mid-68,ybar(r),mid-8,bottom),fill=teal)
        draw.rectangle((mid+8,ybar(b),mid+68,bottom),fill=orange)
        text(mid-38,ybar(r)-12,f'{r:.2f}',22,anchor='mb')
        text(mid+38,ybar(b)-12,f'{b:.2f}',22,anchor='mb')
        text(mid,bottom+15,labels[i],24,anchor='mt')
    text((left+right)/2,670,'Oracle metric / uniform DSM',24,anchor='mt')
    text(920,740,'Population least-squares fits in restricted one-parameter families; no image/video models were trained.',22,muted,anchor='mt')
    text(920,775,'Oracle weighting minimizes the gradient second-moment risk by construction, not the squared mean bias.',22,muted,anchor='mt')
    im.save(OUT/'influence_analytic_figure.png')
    print(json.dumps({'saved':str(OUT/'influence_analytic_results.json'),
                      'figure':str(OUT/'influence_analytic_figure.png'),
                      'max_quadrature_coefficient_delta':max_quadrature_delta,
                      'cubic_bias_squared_ratio':cubic['oracle_query_metric']['mean_student_gradient_bias_squared']/cubic['uniform_DSM']['mean_student_gradient_bias_squared']}))

if __name__=='__main__':
    main()
