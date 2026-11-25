"""Run two batch-64 validations, then resume each arm to G160 on idle GPUs.

Deploy this directory alongside the training code, and launch only after choosing
an appropriate host. Existing incomplete runs are left for inspection.
"""
import argparse
import csv
import fcntl
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from launch import command, gpu_check

CODE = Path(__file__).resolve().parent
ARMS = ('normalized_u', 'raw_angle')
STAGES = tuple((arm, 'validation') for arm in ARMS) + tuple((arm, 'g160') for arm in ARMS)


def load_config(arm, phase):
    suffix = 'batch64_validation' if phase == 'validation' else 'g160'
    config = json.loads((CODE / 'configs' / f'turning_{arm}_2gpu_no_offload_{suffix}.json').read_text())
    if (config['batch'], config['micro_batch'], config['fake_updates'], config['variant']) != (64, 1, 2, 'dmd_only'):
        raise ValueError('Unexpected batch or training configuration')
    if any(config[k] is not False for k in ('fake_offload', 'offload_student', 'offload_student_optimizer', 'benchmark')):
        raise ValueError('This workflow requires full GPU training and real evaluation')
    if config['max_student_updates'] != (2 if phase == 'validation' else 160):
        raise ValueError('Unexpected update count')
    config['turning_profile'] = str(CODE / 'turning_profiles' / f'{arm}_512.json')
    return config


def verify_run(path, target, require_initial_eval=False, world=2):
    """Require completed updates, finite diagnostics, evaluation and full final state."""
    status = json.loads((path / 'status.json').read_text())
    if (status['status'], status['student_updates'], status['fake_updates'], status['D_updates'], status['effective_batch']) != ('complete', target, 2*target, 0, 64):
        raise ValueError(f'Incomplete updates in {path}')
    state = Path(status['training_state'])
    metadata = json.loads((state / 'complete.json').read_text())
    if metadata['world_size'] != world or metadata['progress'] != {'gcount': target, 'fcount': 2*target, 'dcount': 0}:
        raise ValueError('Incomplete training state')
    for rank in range(world):
        for name in ('student', 'fake', 'optimizer_og', 'optimizer_of', 'rng'):
            if not (state / f'{name}_rank{rank}.pt').is_file():
                raise ValueError(f'Missing {name} rank {rank}')
        entries = [json.loads(line) for line in (path / f'rank{rank}.jsonl').read_text().splitlines()]
        students = [x for x in entries if x['phase'] == 'student_done']
        fakes = [x for x in entries if x['phase'] == 'fake']
        # A resumed run contains only the updates made after its saved state.
        expected = target if require_initial_eval else target - 2
        if len(students) != expected or len(fakes) != 2*expected:
            raise ValueError('Missing rank diagnostics')
        for row in students:
            if not all(math.isfinite(row[key]) for key in ('grad_norm', 'dmd_loss', 'pcm_loss', 'middle_dm_loss', 'terminal_dm_loss')):
                raise ValueError('Non-finite student diagnostic')
        if any(not math.isfinite(row['fake_grad_norm']) or not math.isfinite(row['fake_loss']) for row in fakes):
            raise ValueError('Non-finite fake diagnostic')
    for update in ([0, target] if require_initial_eval else [target]):
        matches = list((path / 'eval').glob(f'*_g{update:06d}_*'))
        if not matches:
            raise ValueError(f'Missing evaluation at G{update}')
        for rank in range(world):
            data = json.loads((matches[-1] / f'rank{rank}.json').read_text())
            if data['student_updates'] != update or len(data['samples']) != 8 // world:
                raise ValueError('Incomplete evaluation')
            if any(not (matches[-1] / x['image']).is_file() or not math.isfinite(x['latent_std']) for x in data['samples']):
                raise ValueError('Missing evaluation image or non-finite latent')
    return state


def descendants(pid):
    result, todo = {pid}, [pid]
    while todo:
        current = todo.pop()
        try:
            children = (Path('/proc') / str(current) / 'task' / str(current) / 'children').read_text().split()
        except OSError:
            continue
        for child in map(int, children):
            if child not in result:
                result.add(child)
                todo.append(child)
    return result


def foreign_on(uuids, owned):
    text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=10)
    return [int(r[1]) for r in csv.reader(io.StringIO(text)) if len(r) >= 2 and r[0].strip() in uuids and int(r[1]) not in owned]


def stop_own(p):
    if p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--gpus', required=True, help='Two or four physical GPU indices')
    ap.add_argument('--name', default='turning_batch64_g160_20261027')
    ap.add_argument('--min-free-gib', type=float, default=44)
    ap.add_argument('--shard-teacher', action='store_true')
    args = ap.parse_args()
    root = args.root.resolve()
    gpus = [int(x) for x in args.gpus.split(',')]
    world = len(gpus)
    if world not in (2, 4) or len(set(gpus)) != world or '/' in args.name or '\\' in args.name:
        raise ValueError('Specify two or four different GPUs and a simple output name')
    out = root / 'outputs' / args.name
    out.mkdir(parents=True, exist_ok=True)
    lock = (out / 'workflow.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for required in ('models/sdxl-base-1.0/unet/config.json', 'datasets/parallel_10k_seed20261018/prepared.pt', 'datasets/parallel_10k_seed20261018/manifest.json'):
        if not (root / required).is_file():
            raise FileNotFoundError(root / required)
    # Save space for two validation states, two final states and periodic weights.
    if shutil.disk_usage(root).free < 300 * 1024**3:
        raise RuntimeError('Less than 300 GiB free: inspect checkpoint capacity before launch')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus, NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TORCHDYNAMO_DISABLE='1',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=str(root / 'cache'),
               TMPDIR=str(root / 'tmp'), TOKENIZERS_PARALLELISM='false')
    env.pop('SDXL_SHARED_YIELD_FILE', None)
    (root / 'tmp').mkdir(exist_ok=True)
    states = {}
    for arm, phase in STAGES:
        path = out / f'{arm}_{phase}'
        target = 2 if phase == 'validation' else 160
        if path.exists():
            states[arm, phase] = verify_run(path, target, phase == 'validation', world)
            continue
        if (out / 'USER_STOP').exists():
            raise RuntimeError('User stop requested')
        # Short wait allows utilization to settle after our preceding run exits;
        # a real foreign process or insufficient memory never gets ignored.
        for attempt in range(6):
            try:
                cards = gpu_check(gpus, args.min_free_gib)
                break
            except RuntimeError:
                if attempt == 5:
                    raise
                time.sleep(10)
        uuids = {r[1] for r in cards}
        config = load_config(arm, phase)
        config['shard_teacher'] = args.shard_teacher
        if phase == 'g160':
            config['resume'] = str(states[arm, 'validation'])
        cmd = command(config, root, f'{args.name}/{arm}_{phase}', world)
        with (out / f'{arm}_{phase}.log').open('x') as log:
            p = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            print(json.dumps({'event':'started','arm':arm,'phase':phase,'pid':p.pid,'gpus':cards}), flush=True)
            try:
                while p.poll() is None:
                    foreign = foreign_on(uuids, descendants(p.pid))
                    available = next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
                    if foreign or available < 24 or (out / 'USER_STOP').exists():
                        raise RuntimeError(f'Stopping only our run: foreign={foreign}, available_RAM_GiB={available:.1f}')
                    time.sleep(5)
                if p.returncode:
                    raise RuntimeError(f'{arm} {phase} exited {p.returncode}; inspect its log')
            except BaseException:
                stop_own(p)
                raise
        states[arm, phase] = verify_run(path, target, phase == 'validation', world)
        print(json.dumps({'event':'verified','arm':arm,'phase':phase,'state':str(states[arm, phase])}), flush=True)
    (out / 'workflow_complete.json').write_text(json.dumps({'status':'complete', 'states':{f'{a}_{p}':str(v) for (a,p),v in states.items()}}, indent=2))


if __name__ == '__main__':
    main()
