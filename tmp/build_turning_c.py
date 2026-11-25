"""Create an isolated experimental source tree; original training sources stay intact."""
import bisect
import hashlib
import json
import math
import shutil
import statistics
from pathlib import Path

root = Path(__file__).resolve().parent.parent
out = root / "sdxl_turning_c"
out.mkdir(exist_ok=False)
names = ["train.py", "batching.py", "objectives.py", "pixel.py", "training_state.py", "test_batching.py"]
hashes = {}
for name in names:
    source = root / "sdxl_parallel" / name
    hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copyfile(source, out / name)

source = root / "output/sdxl-crossdistill/hparam_pilot/teacher_turning/result.json"
r = json.loads(source.read_text(encoding="utf-8"))
xs = r["native_t_grid"]
ys = r["modes"]["base_dt10_bf16"]["curves"]["rf_full"]["smoothed_mean"]
assert all(a < b for a, b in zip(xs, xs[1:]))
assert xs[0] < 20 and xs[-1] > 919
def interp(t):
    i = bisect.bisect_right(xs, t)
    a = (t-xs[i-1])/(xs[i]-xs[i-1])
    return (1-a)*ys[i-1]+a*ys[i]
times = list(range(20,920))
rates = [interp(t) for t in times]
offset = .1 * statistics.median(rates)
weights = [(v+offset)**2 for v in rates]
prior = [v/sum(weights) for v in weights]
beta = .3
probabilities = [(1-beta)/len(times)+beta*p for p in prior]
assert all(math.isfinite(p) and p > 0 for p in probabilities)
assert abs(sum(probabilities)-1) < 1e-12
profile = {
    "schema": 1, "name": "C_student_only_turning_mixture", "beta": beta,
    "support": [20,919], "timesteps": times, "probabilities": probabilities,
    "prior_probabilities": prior, "turning_rates": rates, "offset": offset,
    "offset_rule": "0.1 times median smoothed turning rate within DMD support",
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "source": "output/sdxl-crossdistill/hparam_pilot/teacher_turning/result.json",
    "field": "base_dt10_bf16/rf_full/smoothed_mean",
    "interpolation": "linear interpolation on stored native_t_grid; no extrapolation",
    "measure": "probability mass over integer native timesteps, not uniform RF noise",
    "importance_correction": False, "fake_sampler": "unchanged Uniform integers 0..999",
    "student_stage_selection": "unchanged cyclic 2,3,4",
    "mean_t": sum(t*p for t,p in zip(times, probabilities)),
    "mass_t_ge_800": sum(p for t,p in zip(times, probabilities) if t >= 800),
    "max_to_uniform_ratio": max(probabilities)*len(times),
}
(out/"turning_distribution.json").write_text(json.dumps(profile,indent=2),encoding="utf-8")
(out/"base_source_sha256.json").write_text(json.dumps(hashes,indent=2),encoding="utf-8")
p = out/"train.py"
s = p.read_text(encoding="utf-8")
def replace(old,new):
    global s
    assert s.count(old)==1, old
    s=s.replace(old,new)
replace("from pixel import PixelDiscriminator",
        "from pixel import PixelDiscriminator\nfrom turning_sampling import TurningSampler")
replace('    args = arguments()\n',
        '    args = arguments()\n    turning = TurningSampler(Path(__file__).with_name("turning_distribution.json"))\n')
replace('    if rank == 0:\n        write_json(out / "config.json", config)',
        '    config["turning_sampling"] = turning.metadata\n    if rank == 0:\n        write_json(out / "config.json", config)')
replace('("train.py", "batching.py", "pixel.py", "objectives.py", "training_state.py")',
        '("train.py", "batching.py", "pixel.py", "objectives.py", "training_state.py", "turning_sampling.py", "turning_distribution.json", "base_source_sha256.json")')
replace('        randoms = random_inputs(positions, phase, device, seed=args.seed)\n        return indices, condition, real, randoms',
        '        randoms = random_inputs(positions, phase, device, seed=args.seed)\n'
        '        if phase == "student":\n'
        '            chosen = turning.sample(positions, args.seed)\n'
        '            randoms["t"] = torch.tensor(chosen, dtype=torch.long, device=device)\n'
        '            record({"phase": "turning_sample", "g": update, "micro": micro, "timesteps": chosen})\n'
        '        return indices, condition, real, randoms')
p.write_text(s,encoding="utf-8")
p=out/"training_state.py"
s=p.read_text(encoding="utf-8")
s=s.replace('"shard_granularity", "fake_offload", "offload_student", "offload_student_optimizer")',
            '"shard_granularity", "fake_offload", "offload_student", "offload_student_optimizer", "turning_sampling")')
p.write_text(s,encoding="utf-8")
print(json.dumps({k:profile[k] for k in ["name","beta","support","mean_t","mass_t_ge_800","max_to_uniform_ratio"]},indent=2))
