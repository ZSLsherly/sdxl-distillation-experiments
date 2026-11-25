"""Final training checkpoint for the same FSDP topology; periodic weights stay small."""
import json
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def cpu_state(model):
    return {k: local(v).detach().cpu().clone() for k, v in model.state_dict().items()}


def optimizer_state(optimizer):
    state = optimizer.state_dict()
    return {**state, "state": {
        k: {name: local(v).detach().cpu().clone() if torch.is_tensor(v) else v
            for name, v in values.items()} for k, values in state["state"].items()}}


def load_optimizer(optimizer, saved):
    # Adam moments follow each parameter's sharding; its scalar step stays on CPU.
    for group, old_group in zip(optimizer.param_groups, saved["param_groups"]):
        for parameter, key in zip(group["params"], old_group["params"]):
            for name, value in saved["state"].get(key, {}).items():
                if name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    value = value.to(local(parameter).device)
                    if isinstance(parameter, DTensor):
                        value = DTensor.from_local(value, parameter.device_mesh,
                                                   parameter.placements, run_check=False,
                                                   shape=parameter.shape, stride=parameter.stride())
                    saved["state"][key][name] = value
    optimizer.load_state_dict(saved)


def save_training(path, models, optimizers, progress, config):
    path = Path(path)
    rank = dist.get_rank()
    if rank == 0:
        path.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    # Store model/optimizer separately to avoid holding both full CPU copies at once.
    for name, model in models.items():
        if model is not None:
            torch.save(cpu_state(model), path / f"{name}_rank{rank}.pt")
    for name, optimizer in optimizers.items():
        if optimizer is not None:
            torch.save(optimizer_state(optimizer), path / f"optimizer_{name}_rank{rank}.pt")
    torch.save({"progress": progress, "python_rng": random.getstate(),
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
               path / f"rng_rank{rank}.pt")
    dist.barrier()
    if rank == 0:
        (path / "complete.json").write_text(json.dumps({
            "world_size": dist.get_world_size(), "progress": progress, "config": config,
            "kind": "FP32 models, Adam moments, counters and RNG; completed student cycle",
            "resume_requires": "same world size, FSDP layout and training configuration"
        }, indent=2), encoding="utf-8")
    dist.barrier()


def load_training(path, models, optimizers, config):
    path = Path(path)
    metadata = json.loads((path / "complete.json").read_text())
    if metadata["world_size"] != dist.get_world_size():
        raise ValueError("Resume requires the saved world size")
    keys = ("fake_stage_policy", "experiment_modules", "schedule", "pcm_boundary", "dmd_t_max", "pcm_weight",
            "fake_adam_foreach", "student_adam_foreach", "compile_teacher", "batched_teacher_cfg",
            "d_grad_bucket_mb", "control_mode",
            "variant", "batch", "micro_batch", "rollout_batch", "seed", "data_dir",
            "fake_updates", "student_lr", "fake_lr", "d_lr", "gan_weight", "teacher_cfg",
            "shard_granularity", "fake_offload", "offload_student", "offload_student_optimizer")
    for key in keys:
        if metadata["config"].get(key) != config.get(key):
            raise ValueError(f"Resume configuration differs: {key}")
    rank = dist.get_rank()
    for name, model in models.items():
        if model is not None:
            saved = torch.load(path / f"{name}_rank{rank}.pt", map_location="cpu", weights_only=True)
            current = model.state_dict()
            if current.keys() != saved.keys():
                raise ValueError(f"State keys differ: {name}")
            with torch.no_grad():
                for key, value in current.items():
                    local(value).copy_(saved[key])
            del saved
    for name, optimizer in optimizers.items():
        if optimizer is not None:
            load_optimizer(optimizer, torch.load(path / f"optimizer_{name}_rank{rank}.pt",
                                                map_location="cpu", weights_only=True))
    saved = torch.load(path / f"rng_rank{rank}.pt", map_location="cpu", weights_only=True)
    random.setstate(saved["python_rng"])
    torch.set_rng_state(saved["cpu_rng"])
    torch.cuda.set_rng_state(saved["cuda_rng"])
    return saved["progress"]
