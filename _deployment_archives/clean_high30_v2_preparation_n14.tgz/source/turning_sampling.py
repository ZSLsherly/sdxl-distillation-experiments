"""Optional fixed DMD re-noise prior; all other per-sample randomness is untouched."""
import bisect
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


class TurningSampler:
    def __init__(self,path,upper=919):
        self.path=Path(path);raw=self.path.read_bytes();p=json.loads(raw)
        self.times=p['timesteps'];weights=p['probabilities']
        if self.times!=list(range(20,upper+1)) or len(weights)!=len(self.times):
            raise ValueError('Turning prior support must match the student DMD time range')
        if not all(math.isfinite(v) and v>0 for v in weights) or abs(sum(weights)-1)>1e-10:
            raise ValueError('Invalid turning prior probabilities')
        self.cdf=[];total=0.
        for v in weights:total+=v;self.cdf.append(total)
        self.cdf[-1]=1.
        self.profile=p
        self.prior_intervals=p.get('prior_intervals')
        self.interval_rule=p.get('interval_rule')
        if self.interval_rule not in (None,'quota_uniform'):
            raise ValueError('Unknown interval sampling rule')
        if self.interval_rule and not self.prior_intervals:
            raise ValueError('Interval quota requires prior intervals')
        if self.prior_intervals is not None:
            if not self.prior_intervals or any(len(r)!=2 or not self.times[0]<=r[0]<=r[1]<=self.times[-1] for r in self.prior_intervals):
                raise ValueError('Invalid teacher prior intervals')
        self.conditional_cdfs={}
        self.metadata={k:p[k] for k in ['name','metric','beta','support','source_sha256','smoothing','offset_rule','exponent','importance_correction']}
        self.metadata['distribution_sha256']=hashlib.sha256(raw).hexdigest()
        if self.prior_intervals is not None:
            self.metadata['prior_intervals']=self.prior_intervals
        if self.interval_rule:
            self.metadata['interval_rule']=self.interval_rule
        self.metadata['scope']='student DMD re-noise only; fake/PCM/GAN routing unchanged'
        self.metadata['conditional_rule']=('30% uniform inside prior intervals, 70% uniform in their complement within the legal stage support' if self.interval_rule else 'For boundary DM, rebuild 70/30 mixture within each valid stage range; offset is 0.1 times that range median')

    def distribution(self,time_range):
        lo,hi=map(int,time_range)
        if not self.times[0]<=lo<=hi<=self.times[-1]:raise ValueError('Conditional prior outside profiled support')
        times=list(range(lo,hi+1))
        if self.interval_rule=='quota_uniform':
            mask=[any(a<=t<=b for a,b in self.prior_intervals) for t in times]
            n=sum(mask);other=len(times)-n
            if n==0 or other==0:return times,[1/len(times)]*len(times)
            beta=self.profile['beta']
            return times,[beta/n if inside else (1-beta)/other for inside in mask]
        values=self.profile['profile_values'][lo-self.times[0]:hi-self.times[0]+1]
        offset=.1*statistics.median(values);weights=[(v+offset)**self.profile['exponent'] for v in values]
        if self.prior_intervals is not None:
            weights=[w if any(a<=t<=b for a,b in self.prior_intervals) else 0. for t,w in zip(times,weights)]
        total=sum(weights);beta=self.profile['beta']
        if total==0:return times,[1/len(times)]*len(times)
        return times,[(1-beta)/len(times)+beta*v/total for v in weights]

    def sample(self,positions,seed,time_range=None):
        times,cdf=self.times,self.cdf
        if time_range is not None:
            key=tuple(time_range)
            if key not in self.conditional_cdfs:
                times,probabilities=self.distribution(key);cdf=[];total=0.
                for v in probabilities:total+=v;cdf.append(total)
                cdf[-1]=1.;self.conditional_cdfs[key]=(times,cdf)
            times,cdf=self.conditional_cdfs[key]
        return [times[bisect.bisect_right(cdf,random.Random(f"turning:{int(seed)}:{int(pos)}").random())] for pos in positions]

    def apply(self,randoms,positions,phase,seed,time_range=None,include_fake=False):
        if phase=='student' or (phase=='fake' and include_fake):
            randoms['t']=randoms['t'].new_tensor(self.sample(positions,seed,time_range))
        return randoms
