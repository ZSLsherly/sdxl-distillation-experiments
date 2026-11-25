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
        self.conditional_cdfs={}
        self.metadata={k:p[k] for k in ['name','metric','beta','support','source_sha256','smoothing','offset_rule','exponent','importance_correction']}
        self.metadata['distribution_sha256']=hashlib.sha256(raw).hexdigest()
        self.metadata['scope']='student DMD re-noise only; fake/PCM/GAN routing unchanged'
        self.metadata['conditional_rule']='For boundary DM, rebuild 70/30 mixture within each valid stage range; offset is 0.1 times that range median'

    def distribution(self,time_range):
        lo,hi=map(int,time_range)
        if not self.times[0]<=lo<=hi<=self.times[-1]:raise ValueError('Conditional prior outside profiled support')
        times=list(range(lo,hi+1));values=self.profile['profile_values'][lo-self.times[0]:hi-self.times[0]+1]
        offset=.1*statistics.median(values);weights=[(v+offset)**self.profile['exponent'] for v in values]
        total=sum(weights);beta=self.profile['beta']
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
        return [times[bisect.bisect_right(cdf,random.Random(seed+700000+int(pos)).random())] for pos in positions]

    def apply(self,randoms,positions,phase,seed,time_range=None):
        if phase=='student':
            randoms['t']=randoms['t'].new_tensor(self.sample(positions,seed,time_range))
        return randoms
