set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
python - <<'PY'
import os,hashlib,tarfile,subprocess,sys
from pathlib import Path
root=Path(os.environ['SDXL_ROOT'])
needle='    if rank == 0:\n        status = {"status": "stopped" if stopped else "complete"'
insert="""    if rank == 0:
        if args.fake_track_pilot and not stopped:
            from fake_tracking import verify_tracking_probes
            with (out/'rank0.jsonl').open() as stream:
                verify_tracking_probes((json.loads(line) for line in stream),
                                      manifest['holdout_original_indices'][:8],gcount,args.fake_mix_start_g,args.fake_student_mix)
        status = {"status": "stopped" if stopped else "complete" """
insert=insert.rstrip()
assert not (root/'outputs/bc_winner_late_mix_20261102/late_mix').exists() and not list((root/'outputs/flash_gan_g160_20261102').glob('gan_micro*'))
with tarfile.open(root/'tmp/queue_tracking_audit.tar.gz') as t:
    change={n:t.extractfile(n).read().decode('utf-8-sig') for n in ('fake_tracking.py','test_fake_tracking.py')}
for sub,old in [('bc_coco_winner_20261102','8110061fbeb312655a4c72587615bee2a3a69fbd28123fcdb177224c2d9ebdbb'),('flash_gan_g160_20261102','9f153827fb7cf31514a3d43ab98ad06eae59b97449ab2af8af9aa60e5d88488c')]:
    code=root/'tmp'/sub/'code';tr=code/'train.py'
    assert hashlib.sha256(tr.read_bytes()).hexdigest()==old
    assert hashlib.sha256((code/'fake_tracking.py').read_bytes()).hexdigest()=='5fc5e94d13bd5a34cd9823425613f2aebe0b5cb8076e93bd7659521d7604ae57'
    s=tr.read_text();assert s.count(needle)==1
    for n,v in {**change,'train.py':s.replace(needle,insert)}.items():
        compile(v,n,'exec');tmp=code/(n+'.audit');tmp.write_text(v);tmp.replace(code/n)
    out=subprocess.run([sys.executable,'-m','unittest','test_fake_tracking','test_paired_stages','test_interval_quota'],cwd=code,env=dict(os.environ,PYTHONPATH=str(code)),check=True)
    print('Verified',sub,hashlib.sha256(tr.read_bytes()).hexdigest())
PY
