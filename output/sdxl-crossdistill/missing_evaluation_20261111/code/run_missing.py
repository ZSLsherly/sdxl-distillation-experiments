"""Evaluate explicitly transferred G160 checkpoints on the reserved free GPU."""
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path('/data1/shenglin.zeng/coco_missing_20261111')
CODE = Path(__file__).resolve().parent
GPU = 'GPU-b725e1ea-ce0d-1662-ec47-af52200242eb'

def read(path):
    return json.loads(path.read_text())

def children(pid):
    found = {pid}
    pending = [pid]
    while pending:
        p = pending.pop()
        try:
            values = Path(f'/proc/{p}/task/{p}/children').read_text().split()
        except OSError:
            continue
        for value in values:
            child = int(value)
            if child not in found:
                found.add(child)
                pending.append(child)
    return found

def gpu_apps():
    output = subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=10)
    return {int(x) for x in output.splitlines() if x.strip().isdigit()}

def launch_worker(command, logfile):
    row = subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-gpu=memory.free,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=10)
    free, utilization = map(int, next(csv.reader(io.StringIO(row))))
    if gpu_apps() or free < 21000 or utilization > 5:
        raise RuntimeError('Selected GPU is occupied. Recheck before resuming.')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, OMP_NUM_THREADS='4',
               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1')
    with logfile.open('a') as log:
        process = subprocess.Popen(command, cwd=CODE, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                try:
                    conflicts = gpu_apps() - children(process.pid)
                except subprocess.TimeoutExpired:
                    time.sleep(10)
                    continue
                if conflicts:
                    raise RuntimeError(f'Foreign GPU processes appeared: {sorted(conflicts)}')
                time.sleep(10)
            if process.returncode:
                raise RuntimeError(f'Evaluation exited {process.returncode}; retain features for resume')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            raise

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'evaluation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Transfer manifests must be supplied by the source host, then verified here.
        manifests = [ROOT/'transfer_manifest.json', ROOT/'transfer_manifest_old.json']
        assert all(path.is_file() for path in manifests), 'Wait for both transfers to finish'
        for manifest in manifests:
            for relative, (size, digest) in read(manifest).items():
                path = ROOT/relative
                assert path.stat().st_size == size, path
                with path.open('rb') as stream:
                    assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, path
        reference = read(ROOT/'reference/manifest.json')
        assert reference['n'] == len(reference['samples']) == 10000
        assert [s['seed'] for s in reference['samples']] == list(range(20261020, 20271020))
        specs = read(ROOT/'checkpoint_specs.json')
        assert list(specs) == ['A_pcm10_mix1', 'normalized_u_g160', 'raw_angle_g160']
        import numpy as np
        real_ids = []
        for path in sorted((ROOT/'reference/features/real').glob('*.npz')):
            with np.load(path) as chunk:
                real_ids.extend(chunk['indices'].tolist())
                assert chunk['full'].shape == chunk['patch'].shape == (len(chunk['indices']), 2048)
                assert np.isfinite(chunk['full']).all() and np.isfinite(chunk['patch']).all()
        assert real_ids == list(range(10000))
        print('INPUTS_VERIFIED; three G160 checkpoints, fixed COCO10K reference', flush=True)
        for name, relative in specs.items():
            checkpoint = ROOT/relative
            metadata = read(checkpoint/'complete.json')
            assert metadata['student_updates'] == 160
            config = read(checkpoint.parent.parent/'config.json')
            shards = []
            for rank in range(metadata['world_size']):
                path = checkpoint/f'student_rank{rank}.pt'
                with path.open('rb') as stream:
                    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
                shards.append(dict(name=path.name, sha256=digest))
            assert config['schedule'] == [999, 749, 499, 249, -1]
            spec = dict(checkpoint=str(checkpoint), world_size=metadata['world_size'],
                        student_updates=160, config=config, shards=shards)
            result = ROOT/'results'/name
            (result/'features').mkdir(parents=True, exist_ok=True)
            real = result/'features/real'
            if not real.exists():
                real.symlink_to(ROOT/'reference/features/real', target_is_directory=True)
            manifest = dict(reference, models={name: spec})
            path = result/'manifest.json'
            if path.exists():
                assert read(path) == manifest
            else:
                path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
            if (result/'complete.json').exists():
                continue
            arguments = ['--root', str(ROOT), '--out', str(result)]
            launch_worker([sys.executable, '-u', str(CODE/'evaluate.py'), 'worker',
                           *arguments, '--methods', name, '--batch', '2'], result/'worker.log')
            with (result/'metrics.log').open('a') as log:
                subprocess.run([sys.executable, '-u', str(CODE/'evaluate.py'), 'metrics', *arguments],
                               cwd=CODE, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                                                 OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4'),
                               stdout=log, stderr=subprocess.STDOUT, check=True)
            assert read(result/'complete.json')['n_per_model'] == 10000
            print('COMPLETE', name, read(result/'scores.json'), flush=True)

if __name__ == '__main__':
    main()
