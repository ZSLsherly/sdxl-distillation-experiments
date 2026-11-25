"""Explicit objective/exposure policy, independent of CUDA and model code."""
VARIANTS=('dmd','gan4','joint4','joint34_matched','joint34_full')
def generator_stage(update):return 2+update%3
def gan_active(variant,update):
    stage=generator_stage(update)
    if variant=='dmd':return False
    if variant in ('gan4','joint4'):return stage==4
    if variant=='joint34_full':return stage in (3,4)
    if variant=='joint34_matched':
        cycle=update//3
        return (stage==3 and cycle%2==0) or (stage==4 and cycle%2==1)
    raise ValueError(variant)
def dmd_active(variant,update):return not(variant=='gan4' and generator_stage(update)==4)
def discriminator_stage(variant,update):
    if variant=='dmd':return None
    if variant in ('gan4','joint4'):return 4
    if variant in ('joint34_matched','joint34_full'):return 3+update%2
    raise ValueError(variant)
def variant_for_iteration(variant,g_update,preflight):
    # One shared model allocation exercises no-D, stage-3 GAN, and pure GAN4.
    return ('dmd','joint34_full','gan4')[g_update%3] if preflight else variant
def check_policy():
    for v in VARIANTS:
        stages=[generator_stage(g) for g in range(24) if gan_active(v,g)]
        expected={'dmd':[],'gan4':[4]*8,'joint4':[4]*8,'joint34_matched':[3]*4+[4]*4,'joint34_full':[3]*8+[4]*8}[v]
        assert sorted(stages)==expected
        d=[discriminator_stage(v,k) for k in range(120)]
        assert set(stages).issubset(set(d))
        if v.startswith('joint34'):assert d.count(3)==d.count(4)==60
        if v=='dmd':assert all(s is None for s in d)
    assert sum(not dmd_active('gan4',g) for g in range(24))==8

if __name__=='__main__':
    check_policy();print('PASS: matched GAN exposures, stage-matched D support, pure DMD bypass, GAN4 replacement')
