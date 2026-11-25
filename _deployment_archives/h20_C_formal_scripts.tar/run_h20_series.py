import json,os,subprocess,sys,time
from pathlib import Path
from launch import command,gpu_check
from turning_batch64_workflow import descendants,foreign_on,stop_own

root=Path(os.environ['SDXL_ROOT'])
outputs=root/'outputs'
cfg=json.loads(Path('config.json').read_text())
assert cfg['batch']==64 and cfg['micro_batch']==16 and cfg['fake_mix_start_g']==1 and cfg['gan_weight']==.005
def execute(conf,name):
    gpu=gpu_check([0],90)[0][1]
    if (outputs/name).exists():raise FileExistsError(name)
    with open(name+'.log','x',buffering=1) as log:
        p=subprocess.Popen(command(conf,root,name,1),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while p.poll() is None:
                if foreign_on({gpu},descendants(p.pid)):raise RuntimeError('GPU conflict')
                time.sleep(4)
        except BaseException:
            stop_own(p)
            raise
    return p.returncode
ref=json.loads((outputs/'h20_gan_mb16_fp32sam4/benchmark.json').read_text())
baseline=sum(ref['cycle_seconds'][1:])/2
name='h20_fp32sam4_vae2_probe'
formal='h20_C_mix1_terminal_GAN_G160'
assert not (outputs/formal).exists()
gpu_check([0],90)
subprocess.run([sys.executable,'vae_compare.py'],check=True)
pick=1
selection={'baseline_seconds':baseline,'probe':json.loads(Path('vae_compare.json').read_text()),'vae_batch':1}
if selection['probe']['pass']:
    rc=execute(dict(cfg,vae_batch=2),name)
    if rc==0:
        r=json.loads((outputs/name/'benchmark.json').read_text())
        assert (r['student_updates'],r['fake_updates'],r['D_updates'])==(3,6,3)
        seconds=sum(r['cycle_seconds'][1:])/2
        if seconds < .98*baseline and r['peak_reserved_GiB']<90: pick=2
        selection.update(seconds=seconds,peak_reserved_GiB=r['peak_reserved_GiB'])
    elif 'out of memory' in Path(name+'.log').read_text().lower():
        selection['candidate']='OOM'
    else:raise RuntimeError('VAE2 training benchmark failed; inspect log')
selection['vae_batch']=pick
Path('selection.json').write_text(json.dumps(selection,indent=2))
print('Selected',selection,flush=True)
conf=dict(cfg,benchmark=False,vae_batch=pick,max_student_updates=160)
assert conf['checkpoint_mode']=='final-weights' and conf['save_every_updates']==0
rc=execute(conf,formal)
if rc:raise SystemExit(rc)
r=json.loads((outputs/formal/'status.json').read_text())
assert (r['student_updates'],r['fake_updates'],r['D_updates'])==(160,320,160),r
print('Completed',r,flush=True)
