"""Deterministic timestep sampling on a separate stream; no changes to latent RNG."""
import bisect
import hashlib
import json
import math
import random
from pathlib import Path

class TurningSampler:
    def __init__(self, path):
        path = Path(path)
        raw = path.read_bytes()
        profile = json.loads(raw)
        self.times = profile["timesteps"]
        probabilities = profile["probabilities"]
        if self.times != list(range(20,920)) or len(probabilities) != len(self.times):
            raise ValueError("Unexpected timestep support")
        if not all(math.isfinite(p) and p > 0 for p in probabilities):
            raise ValueError("Invalid probabilities")
        if abs(sum(probabilities)-1) > 1e-10:
            raise ValueError("Probabilities must sum to one")
        self.cdf = []
        value = 0.
        for p in probabilities:
            value += p
            self.cdf.append(value)
        self.cdf[-1] = 1.
        self.metadata = {k:profile[k] for k in [
            "name","beta","support","offset","offset_rule","source_sha256","field",
            "interpolation","measure","importance_correction","fake_sampler","student_stage_selection",
            "mean_t","mass_t_ge_800","max_to_uniform_ratio"]}
        self.metadata["distribution_sha256"] = hashlib.sha256(raw).hexdigest()
        self.metadata["rng"] = "Python Random(seed + 700000 + global_sample_position), separate from original CUDA RNG"

    def sample(self, positions, seed):
        return [self.times[bisect.bisect_right(self.cdf, random.Random(seed+700000+int(p)).random())]
                for p in positions]
