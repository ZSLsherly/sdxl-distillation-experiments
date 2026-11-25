"""Real batched SDXL TM/DMD/PixelGAN training; compatible with torch 2.5 FSDP2.

Example: torchrun --standalone --nproc_per_node=2 sdxl_parallel/train.py \
    --run gan_stage_parallel/joint4_mb2 --variant joint4 --batch 64 --micro-batch 2
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import CPUOffloadPolicy, MixedPrecisionPolicy, fully_shard
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformers.transformer_2d import Transformer2DModel

from batching import (BatchPlan, cfg_prediction, conditions, decode_tiles, flip_images,
                      r1_penalty, random_inputs, rollout_chunks, slice_batch)
from objectives import VPSchedule, discriminator_loss, dmd_surrogate
from pixel import PixelDiscriminator
from training_state import save_training, load_training
from runtime import average_gradients, TraceWindow, schedule_settings


def stamp():
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%dT%H%M%S%z")


def local(value):
    return value.to_local() if hasattr(value, "to_local") else value


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/data5/shenglin.zeng/sdxl_crossdistill"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--resume", help="Completed final training-state directory; same GPU count and settings")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/gan_v3"),
                        help="Prepared dataset directory, absolute or relative to root")
    parser.add_argument("--variant", choices=("joint4", "gan4", "joint34", "dmd_only"), required=True)
    parser.add_argument("--schedule", type=int, nargs=4, default=[999,919,499,199])
    parser.add_argument("--pcm-boundary", type=int, help="Default: second sampling point")
    parser.add_argument("--dmd-t-max", type=int, default=919, help="DMD re-noise upper bound; independent of inference schedule")
    parser.add_argument("--pcm-weight", type=float, default=10)
    parser.add_argument("--d-grad-bucket-mb", type=float, default=16, help="0 restores per-parameter reductions")
    parser.add_argument("--fake-adam-foreach", action="store_true", help="Opt-in: validate DTensor/device support before use")
    parser.add_argument("--student-adam-foreach", action="store_true")
    parser.add_argument("--compile-teacher", action="store_true", help="Opt-in frozen teacher only; include compile warmup in cost")
    parser.add_argument("--control-mode", choices=("cycle", "legacy"), default="cycle")
    parser.add_argument("--trace-cycles", type=int, default=0, help="Rank0 PyTorch trace; do not use trace runs for throughput")
    parser.add_argument("--benchmark-cycles", type=int, default=9)
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--batch", type=int, default=64, help="Effective global batch per optimizer update")
    parser.add_argument("--micro-batch", type=int, default=2, help="Images processed together on EACH GPU")
    parser.add_argument("--vae-batch", type=int, default=4,
                        help="Maximum parallel VAE tiles during GAN backward; UNet batch is unchanged")
    parser.add_argument("--rollout-batch", type=int, default=8,
                        help="No-grad sample-generation batch per GPU, independent of training micro-batch")
    parser.add_argument("--batched-teacher-cfg", action="store_true",
                        help="Optional: combine teacher unconditional/conditional calls; needs more activation memory")
    parser.add_argument("--shard-granularity", choices=("fine", "block"), default="fine",
                        help="Optional block mode reduces FSDP call count but may increase peak memory")
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--max-student-updates", type=int)
    parser.add_argument("--student-lr", type=float, default=1e-6)
    parser.add_argument("--fake-lr", type=float, default=1e-6)
    parser.add_argument("--d-lr", type=float, default=1e-4)
    parser.add_argument("--fake-updates", type=int, default=5)
    parser.add_argument("--gan-weight", type=float, default=0.005274229682981968)
    parser.add_argument("--teacher-cfg", type=float, default=8)
    parser.add_argument("--fake-offload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--offload-student", action="store_true")
    parser.add_argument("--offload-student-optimizer", action="store_true",
                        help="Keep Adam moments in CPU RAM between student updates; parameters remain on GPU")
    parser.add_argument("--eval-every-samples", type=int, default=128)
    parser.add_argument("--snapshot-seconds", type=int, default=1800)
    parser.add_argument("--save-every-updates", type=int, default=0,
                        help="Save committed student weights every N optimizer updates; 0 uses time only")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--profile", action="store_true", help="CUDA-synchronized subphase timings; adds synchronization")
    parser.add_argument("--gradient-probe-every", type=int, default=0,
                        help="Optional extra GAN backward on first microbatch every N GAN updates; 0 disables")
    parser.add_argument("--benchmark", action="store_true",
                        help="Real stage cycle and fake:G ratio, no eval/save; separate from formal training")
    parser.add_argument("--benchmark-batch", type=int, default=64,
                        help="Fixed global probe batch, so different micro/rollout batches process equal work")
    args = parser.parse_args()
    if args.fake_updates < 1 or args.eval_every_samples < 1 or args.snapshot_seconds < 1 or args.hours <= 0:
        parser.error("Update counts, intervals, and hours must be positive")
    if args.max_student_updates is not None and args.max_student_updates < 1:
        parser.error("max-student-updates must be positive")
    if args.gradient_probe_every < 0:
        parser.error("gradient-probe-every must be nonnegative")
    if args.rollout_batch < 1:
        parser.error("rollout-batch must be positive")
    if args.save_every_updates < 0:
        parser.error("save-every-updates must be nonnegative")
    if args.pcm_weight < 0 or args.trace_cycles < 0 or args.d_grad_bucket_mb < 0:
        parser.error("Loss weights, trace cycles and bucket size must be nonnegative")
    if not 0 <= args.benchmark_warmup < args.benchmark_cycles:
        parser.error("Benchmark requires at least one measured cycle")
    try:
        schedule_settings(args.schedule, args.dmd_t_max, args.pcm_boundary)
    except ValueError as error:
        parser.error(str(error))
    if args.compile_teacher and os.environ.get("TORCHDYNAMO_DISABLE") == "1":
        parser.error("Unset TORCHDYNAMO_DISABLE to use --compile-teacher")
    return args


def main():
    args = arguments()
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    gpu = int(os.environ["LOCAL_RANK"])
    requested = BatchPlan(args.batch, world, args.micro_batch)
    plan = BatchPlan(args.benchmark_batch, world, args.micro_batch) if args.benchmark else requested
    torch.cuda.set_device(gpu)
    device = torch.device("cuda", gpu)
    torch.set_num_threads(4)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    out = args.root / "outputs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    if (out / f"rank{rank}.jsonl").exists():
        raise FileExistsError(f"Use a fresh run name: {out}")
    (out / f"pid_rank{rank}").write_text(str(os.getpid()))
    base = args.root / "models/sdxl-base-1.0"
    data_dir = args.data_dir if args.data_dir.is_absolute() else args.root / args.data_dir
    data = torch.load(data_dir / "prepared.pt", weights_only=True, mmap=True, map_location="cpu")
    manifest = json.loads((data_dir / "manifest.json").read_text())
    order = random.Random(args.seed).sample(range(data["train_count"]), data["train_count"])
    mesh = init_device_mesh("cuda", (world,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True)
    torch.manual_seed(args.seed)

    def load_model(train, offload=False):
        model = UNet2DConditionModel.from_pretrained(
            base / "unet", variant="fp16", torch_dtype=torch.float32 if train else torch.bfloat16,
            local_files_only=True).to(device)
        model.requires_grad_(train).train(train)
        if train:
            model.enable_gradient_checkpointing()
            extra = {"offload_policy": CPUOffloadPolicy()} if offload else {}
            shard_types = (ResnetBlock2D, Transformer2DModel, Downsample2D, Upsample2D)
            if args.shard_granularity == "fine":
                shard_types += (BasicTransformerBlock,)
            # Fine preserves v4's layout; block mode removes inner transformer
            # groups. Neither mode keeps full gradients across microbatches.
            for child in reversed(list(model.modules())):
                if isinstance(child, shard_types):
                    fully_shard(child, mesh=mesh, mp_policy=policy, **extra)
            fully_shard(model, mesh=mesh, mp_policy=policy, **extra)
        return model

    print("LOAD", rank, "micro_batch", plan.micro_batch, "accumulation", plan.accumulation, flush=True)
    student = load_model(True, args.offload_student)
    fake = load_model(True, args.fake_offload)
    teacher = load_model(False)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", variant="fp16", torch_dtype=torch.float32,
                                      local_files_only=True).to(device).eval().requires_grad_(False)
    # Checkpoint individual decoder blocks as well as the outer tile call:
    # recomputing the whole decoder alone still retains all its inner activations.
    vae.enable_gradient_checkpointing()
    torch.manual_seed(args.seed + 7000)
    disc = None
    if args.variant != "dmd_only":
        disc = PixelDiscriminator(args.root / "models/dinov2-small").to(device)
        disc.set_heads(False)
    og = torch.optim.AdamW(student.parameters(), lr=args.student_lr, betas=(.9, .999), weight_decay=.01, foreach=args.student_adam_foreach)
    of = torch.optim.AdamW(fake.parameters(), lr=args.fake_lr, betas=(.9, .999), weight_decay=.01, foreach=args.fake_adam_foreach)
    od = torch.optim.AdamW(disc.trainable(), lr=args.d_lr, betas=(0, .99), weight_decay=0, foreach=False) if disc is not None else None
    scheduler = VPSchedule(DDIMScheduler.from_pretrained(base, subfolder="scheduler", local_files_only=True).alphas_cumprod.to(device))
    knots, pcm_boundary, dmd_t_max = schedule_settings(args.schedule, args.dmd_t_max, args.pcm_boundary)
    if args.compile_teacher:
        teacher.forward = torch.compile(teacher.forward, mode="default", fullgraph=False)
    config = {**vars(args), "root": str(args.root), "data_dir": str(data_dir), "world_size": world, "effective_batch": plan.global_batch,
              "accumulation": plan.accumulation, "microbatch_per_gpu": plan.micro_batch,
              "requested_training_batch": requested.global_batch, "schedule": knots,
              "pcm_boundary": pcm_boundary, "dmd_t_max": dmd_t_max,
              "pcm_t_range": [pcm_boundary + 21, 999], "student_dmd_t_range": [20,dmd_t_max],
              "runtime_versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                   "device": torch.cuda.get_device_name(gpu)},
              "student_cfg": 1, "train_count": data["train_count"],
              "initialization": f"full training state from {args.resume}" if args.resume else "untouched SDXL Base; independent student/fake/D per run",
              "checkpoint_kind": "periodic BF16 student weights; final FP32 student/fake/D + optimizers/counters/RNG",
              "spectral_norm_policy": "20 initialization iterations; one refresh per D optimizer step, fixed across microbatches",
              "fake_rollout_batches_per_update_per_rank": math.ceil((plan.global_batch // world) / args.rollout_batch),
              "rollout_cache": "one local effective fake batch; GPU latents only plus its existing batch inputs; discarded before next student update",
              "batch_semantics": "mean loss / accumulation, followed by FSDP mean over ranks",
              "augmentation": "independent per-image crop/flip; sample RNG does not depend on batch packing",
              "gan_stages": [] if disc is None else ([3, 4] if args.variant == "joint34" else [4]),
              "discriminator_enabled": disc is not None,
              "benchmark_protocol": "same fake:G ratio and stages 2/3/4 as training; excludes evaluation/checkpoint IO; warmup recorded separately" if args.benchmark else None}
    if rank == 0:
        write_json(out / "config.json", config)
        write_json(out / "data_manifest.json", manifest)
        source = out / "source"
        source.mkdir(exist_ok=True)
        hashes = {}
        for name in ("train.py", "batching.py", "pixel.py", "objectives.py", "training_state.py", "runtime.py"):
            file = Path(__file__).with_name(name)
            shutil.copyfile(file, source / name)
            hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        write_json(out / "source_sha256.json", hashes)
    log = (out / f"rank{rank}.jsonl").open("w", encoding="utf-8")
    gcount = fcount = dcount = 0
    models = {"student": student, "fake": fake, "discriminator": disc}
    optimizers = {"og": og, "of": of, "od": od}
    if args.resume:
        progress = load_training(args.resume, models, optimizers, config)
        gcount, fcount, dcount = (progress[k] for k in ("gcount", "fcount", "dcount"))
        if args.offload_student_optimizer:
            for state in og.state.values():
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in state:
                        state[key] = state[key].cpu()
        print("RESUMED", rank, args.resume, progress, flush=True)
    last_saved_gcount = -1
    start = last_snapshot = last_status = time.monotonic()
    phase_times = defaultdict(float)
    benchmark_cycles = []
    stop_requested = False
    torch.cuda.reset_peak_memory_stats()
    trace = TraceWindow(args.trace_cycles, out / "trace_rank0.json", rank == 0)

    @contextmanager
    def measure(name):
        if args.profile:
            torch.cuda.synchronize()
            tick = time.monotonic()
        with trace.region(name):
            yield
        if args.profile:
            torch.cuda.synchronize()
            phase_times[name] += time.monotonic() - tick

    def predict(model, x, t, condition, negative=False):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model(x, t,
                         encoder_hidden_states=condition["negative_prompt_embeds" if negative else "prompt_embeds"],
                         added_cond_kwargs={"text_embeds": condition["negative_pooled_prompt_embeds" if negative else "pooled_prompt_embeds"],
                                            "time_ids": condition["time_ids"]}, return_dict=False)[0].float()

    def guided(x, t, condition):
        return cfg_prediction(predict, teacher, x, t, condition, args.teacher_cfg, args.batched_teacher_cfg)

    def rollout(condition, noise, stage=4, grad=False):
        x = noise
        batch = x.shape[0]
        for i in range(stage):
            t = torch.full((batch,), knots[i], device=device, dtype=torch.long)
            with torch.enable_grad() if grad and i == stage - 1 else torch.no_grad():
                eps = predict(student, x, t, condition)
                x = scheduler.jump(x, eps, t, knots[i + 1] if i < stage - 1 else -1)
            if i < stage - 1:
                x = x.detach()
        return x

    def training_batch(update, micro, phase):
        positions = plan.positions(update, micro, rank)
        indices = [order[pos % len(order)] for pos in positions]
        condition = conditions(data, indices, device)
        real = data["latents"].index_select(0, torch.tensor(indices)).to(device, dtype=torch.float32)
        randoms = random_inputs(positions, phase, device, seed=args.seed,
                               pcm_boundary=pcm_boundary, dmd_t_max=dmd_t_max)
        return indices, condition, real, randoms

    def generate_fake_batch(update):
        # Student weights are fixed for the entire fake/D update. Generate its
        # samples with larger no-grad batches, then slice this cache into the
        # unchanged backward microbatches. No samples are reused across updates.
        positions = plan.rank_positions(update, rank)
        indices = [order[pos % len(order)] for pos in positions]
        condition = conditions(data, indices, device)
        randoms = random_inputs(positions, "fake", device, seed=args.seed)
        generated = rollout_chunks(rollout, condition, randoms["noise"], args.rollout_batch)
        # Initial rollout noise is not needed once all final latents are cached.
        del randoms["noise"]
        real = data["latents"].index_select(0, torch.tensor(indices)).to(device, dtype=torch.float32)
        return indices, condition, real, randoms, generated

    def clip_sharded(model):
        gradients = [local(p.grad) for p in model.parameters() if p.grad is not None]
        if not gradients:
            raise RuntimeError("Expected gradients")
        # fake gradients live on CPU: reduce there first, then transfer ONE
        # scalar, instead of one CUDA copy/synchronization per parameter.
        norms = torch.stack(torch._foreach_norm(gradients, 2))
        total = norms.double().square().sum().to(device)
        dist.all_reduce(total)
        norm = total.sqrt().item()
        if not math.isfinite(norm):
            raise FloatingPointError(f"Non-finite gradient norm: {norm}")
        torch._foreach_mul_(gradients, min(1., 10. / (norm + 1e-12)))
        return norm

    def average_disc_gradients():
        # D is replicated. The only reduction is at the optimizer boundary.
        return average_gradients(disc.trainable(), args.d_grad_bucket_mb)

    def summarize(values, names):
        value = torch.stack(values).mean(0)
        dist.all_reduce(value)
        value.div_(world)
        return dict(zip(names, value.cpu().tolist()))

    def record(row):
        row.update(timestamp=stamp(), effective_batch=plan.global_batch,
                   micro_batch=plan.micro_batch, accumulation=plan.accumulation)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print("UPDATE", rank, row["phase"], row.get("g", row.get("fake_update")), row.get("seconds"), flush=True)

    def snapshot(reason, phase):
        nonlocal last_snapshot, last_saved_gcount
        names = [stamp() if rank == 0 else None]
        dist.broadcast_object_list(names, src=0)
        dest = out / "weights" / f"{names[0]}_g{gcount:06d}_{reason}"
        dest.mkdir(parents=True, exist_ok=True)
        state = {key: local(value).detach().to(torch.bfloat16).cpu() for key, value in student.state_dict().items()}
        torch.save(state, dest / f"student_rank{rank}.pt")
        del state
        write_json(dest / f"shapes_rank{rank}.json", {
            key: {"shape": list(value.shape), "placements": [str(p) for p in value.placements] if hasattr(value, "placements") else []}
            for key, value in student.state_dict().items()})
        dist.barrier()
        if rank == 0:
            write_json(dest / "complete.json", {"timestamp": names[0], "student_updates": gcount,
                       "student_samples": gcount * plan.global_batch, "fake_updates": fcount,
                       "phase": phase, "reason": reason, "world_size": world,
                       "note": "Committed student weights only; partial gradients and optimizer state are not saved"})
        last_snapshot = time.monotonic()
        last_saved_gcount = gcount

    def control(phase, micro):
        nonlocal last_status, stop_requested
        now = time.monotonic()
        # A stop was already acted on only at cycle_start in the old code.
        # Avoid redundant GPU broadcasts/synchronizations inside every microbatch.
        # Time-only snapshots also move to the completed-cycle boundary.
        if args.control_mode == "cycle" and phase != "cycle_start":
            if rank == 0 and now - last_status >= 30:
                write_json(out / "status.json", {"status":"running", "timestamp":stamp(),
                    "phase":phase, "student_updates":gcount, "fake_updates":fcount,
                    "D_updates":dcount, "effective_batch":plan.global_batch,
                    "micro_finished":micro, "micro_total":plan.accumulation,
                    "elapsed_seconds":now-start})
                last_status = now
            return False
        flags = [0, 0]
        if rank == 0:
            flags[0] = int(not args.benchmark and not args.save_every_updates and now - last_snapshot >= args.snapshot_seconds)
            flags[1] = int(now - start >= args.hours * 3600 or (out / "STOP_REQUEST").exists() or (out.parent / "STOP_REQUEST").exists())
        tensor = torch.tensor(flags, device=device)
        dist.broadcast(tensor, src=0)
        save, stop = tensor.cpu().tolist()
        stop_requested = stop_requested or bool(stop)
        if save:
            snapshot("half_hour", phase)
        if rank == 0 and (now - last_status >= 30 or micro == 0):
            write_json(out / "status.json", {"status": "running", "timestamp": stamp(), "phase": phase,
                       "student_updates": gcount, "student_samples": gcount * plan.global_batch,
                       "fake_updates": fcount, "D_updates": dcount, "micro_finished": micro,
                       "micro_total": plan.accumulation, "microbatch_per_gpu": plan.micro_batch,
                       "effective_batch": plan.global_batch, "elapsed_seconds": now - start,
                       "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3})
            last_status = now
        # Finish the current cycle so the final state has no partial gradients
        # or half-finished fake:G schedule. Stop latency is at most one cycle.
        return stop_requested and phase == "cycle_start"

    @torch.no_grad()
    def evaluate(tag):
        names = [stamp() if rank == 0 else None]
        dist.broadcast_object_list(names, src=0)
        dest = out / "eval" / f"{names[0]}_g{gcount:06d}_{tag}"
        dest.mkdir(parents=True, exist_ok=True)
        heldout = [data["original_indices"].index(i) for i in manifest["eval_original_indices"]]
        all_cases = [("train", i) for i in order[:4]] + [("holdout", i) for i in heldout[:4]]
        if len(all_cases) % world:
            raise ValueError("Evaluation case count must divide world_size for FSDP")
        cases = all_cases[rank::world]
        rows = []
        from PIL import Image
        for offset in range(0, len(cases), plan.micro_batch):
            selected = cases[offset:offset + plan.micro_batch]
            c = conditions(data, [i for _, i in selected], device)
            noise = torch.stack([torch.randn((4, 128, 128), device=device,
                                generator=torch.Generator(device=device).manual_seed(102)) for _ in selected])
            generated = rollout(c, noise)
            if not torch.isfinite(generated).all():
                raise FloatingPointError("Non-finite evaluation output")
            # Training networks are batched. Full 1024px FP32 VAE evaluation
            # stays image-by-image to avoid an unrelated full-frame VAE OOM.
            for j, (split, idx) in enumerate(selected):
                x = generated[j:j + 1]
                rgb = vae.decode(x / vae.config.scaling_factor, return_dict=False)[0]
                image = ((rgb[0] / 2 + .5).clamp(0, 1).permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
                filename = f'{split}_original{data["original_indices"][idx]}_seed102.png'
                Image.fromarray(image).save(dest / filename)
                row = {"split": split, "original_index": data["original_indices"][idx], "prompt": data["prompts"][idx],
                       "image": filename, "seed": 102, "latent_std": x.std().item()}
                # Random spectral-norm buffers have not been iterated at g=0;
                # do not misreport their raw initial logits as a quality score.
                if dcount:
                    real = data["latents"][idx:idx + 1].to(device, dtype=torch.float32)
                    pair = torch.cat([real, x])
                    tile = decode_tiles(vae, pair, [(40, 40)] * 2)
                    parts = disc(tile, c["pooled_prompt_embeds"][j:j + 1].expand(2, -1), return_parts=True)
                    row["D_real_fake"] = {key: parts[key].tolist() for key in ("total", "dino", "rgb")}
                rows.append(row)
        write_json(dest / f"rank{rank}.json", {"student_updates": gcount,
                   "student_samples": gcount * plan.global_batch, "samples": rows})
        dist.barrier()
        print("EVALUATED", rank, gcount, flush=True)

    if not args.benchmark and not args.resume:
        evaluate("initial")
    limit = args.benchmark_cycles if args.benchmark else (args.max_student_updates or 10**9)
    stopped = False
    trace.start()
    while gcount < limit:
        if control("cycle_start", 0):
            stopped = True
            break
        torch.cuda.synchronize()
        cycle_start = time.monotonic()
        stage = 2 + gcount % 3
        repeats = args.fake_updates
        for _ in range(repeats):
            tick = time.monotonic()
            of.zero_grad(set_to_none=True)
            if disc is not None:
                od.zero_grad(set_to_none=True)
                disc.set_heads(True)
            stats, originals = [], []
            with measure("fake_student_rollout"):
                cached_indices, cached_condition, cached_real, cached_randoms, cached_generated = generate_fake_batch(fcount)
            for micro in range(plan.accumulation):
                begin, end = micro * plan.micro_batch, (micro + 1) * plan.micro_batch
                indices = cached_indices[begin:end]
                condition = slice_batch(cached_condition, begin, end)
                real = cached_real[begin:end]
                rnd = slice_batch(cached_randoms, begin, end)
                generated = cached_generated[begin:end]
                originals.extend(data["original_indices"][i] for i in indices)
                dvalues = torch.empty(0, device=device)
                if disc is not None:
                    with measure("D_decode"), torch.no_grad():
                        real_rgb = flip_images(decode_tiles(vae, real, rnd["crops"]), rnd["flips"])
                        fake_rgb = flip_images(decode_tiles(vae, generated, rnd["crops"]), rnd["flips"])
                    with measure("D_forward_backward"):
                        r1_on = dcount % 16 == 15
                        real_rgb.requires_grad_(r1_on)
                        pr = disc(real_rgb, condition["pooled_prompt_embeds"], return_parts=True)
                        pf = disc(fake_rgb, condition["pooled_prompt_embeds"], return_parts=True)
                        hinge = discriminator_loss(pr["total"], pf["total"])
                        r1 = r1_penalty(pr["total"], real_rgb) if r1_on else torch.zeros((), device=device)
                        dloss = hinge + .05 * 16 * r1
                        dvalues = torch.stack([hinge, r1, pr["total"].mean(), pf["total"].mean(),
                                              pr["dino"].mean(), pf["dino"].mean(), pr["rgb"].mean(), pf["rgb"].mean()]).detach()
                        (dloss / plan.accumulation).backward()
                        del real_rgb, fake_rgb, pr, pf, hinge, r1, dloss
                with measure("fake_forward_backward"):
                    xt = scheduler.noise(generated, rnd["eps"], rnd["t"])
                    floss = (predict(fake, xt, rnd["t"], condition) - rnd["eps"]).square().mean()
                    (floss / plan.accumulation).backward()
                    stats.append(torch.cat([floss.detach().reshape(1), dvalues]))
                    del xt, floss, generated, real, rnd, condition, dvalues
                if control("fake_D" if disc is not None else "fake", micro + 1):
                    stopped = True
                    break
            del cached_indices, cached_condition, cached_real, cached_randoms, cached_generated
            if stopped:
                break
            if disc is not None:
                with measure("D_optimizer"):
                    d_collectives = average_disc_gradients()
                    dnorm = torch.nn.utils.clip_grad_norm_(disc.trainable(), 10, error_if_nonfinite=True).item()
                    od.step()
                    od.zero_grad(set_to_none=True)
                    disc.set_heads(False)
            with measure("fake_optimizer"):
                fnorm = clip_sharded(fake)
                of.step()
                of.zero_grad(set_to_none=True)
            fcount += 1
            dcount += int(disc is not None)
            torch.cuda.synchronize()
            dmetrics = ["D_hinge", "R1_raw", "D_real", "D_fake", "D_dino_real", "D_dino_fake", "D_rgb_real", "D_rgb_fake"] if disc is not None else []
            record({"phase": "fake_D" if disc is not None else "fake", "fake_update": fcount, "D_update": dcount,
                    "seconds": time.monotonic() - tick, "fake_grad_norm": fnorm,
                    "rollout_batch": args.rollout_batch,
                    "rollout_batches_per_rank": math.ceil((plan.global_batch // world) / args.rollout_batch),
                    "original_indices_rank": originals,
                    **({"D_grad_norm": dnorm, "R1_active": r1_on, "D_collectives":d_collectives} if disc is not None else {}),
                    **summarize(stats, ["fake_loss"] + dmetrics)})
        if stopped:
            break
        og.zero_grad(set_to_none=True)
        tick = time.monotonic()
        stats, originals = [], []
        use_dmd = not (args.variant == "gan4" and stage == 4)
        use_gan = disc is not None and (stage == 4 or (args.variant == "joint34" and stage == 3))
        for micro in range(plan.accumulation):
            indices, condition, real, rnd = training_batch(gcount, micro, "student")
            originals.extend(data["original_indices"][i] for i in indices)
            with measure("G_rollout"):
                generated = rollout(condition, rnd["noise"], stage, True)
            with measure("G_low_objective_backward"):
                loss = torch.zeros((), device=device)
                dmd_value = gan_value = torch.zeros((), device=device)
                if use_dmd:
                    xt = scheduler.noise(generated, rnd["eps"], rnd["t"])
                    with torch.no_grad():
                        f0 = scheduler.clean(xt, predict(fake, xt, rnd["t"], condition), rnd["t"])
                        t0 = scheduler.clean(xt, guided(xt, rnd["t"], condition), rnd["t"])
                    dmd, direction = dmd_surrogate(generated, f0, t0)
                    loss = loss + dmd
                    dmd_value = dmd.detach()
                    del xt, f0, t0, direction, dmd
                if use_gan:
                    rgb = decode_tiles(vae, generated, rnd["crops"], grad=True, chunk_size=args.vae_batch)
                    adversarial = -disc(rgb, condition["pooled_prompt_embeds"]).mean()
                    if args.gradient_probe_every and (gcount // 3) % args.gradient_probe_every == 0 and micro == 0:
                        derivative = torch.autograd.grad(adversarial, generated, retain_graph=True)[0]
                        # Undo mean-over-batch scaling for a batch-size-independent diagnostic.
                        rms = derivative.square().flatten(1).mean(1).sqrt() * plan.micro_batch
                        record({"phase": "GAN_gradient_probe", "g": gcount, "per_image_latent_rms": rms.tolist()})
                        del derivative, rms
                    loss = loss + args.gan_weight * adversarial
                    gan_value = adversarial.detach()
                (loss / plan.accumulation).backward()
                del generated, loss
                if use_gan:
                    del rgb, adversarial
            with measure("G_PCM_backward"):
                th = rnd["pcm_t"]
                xh = scheduler.noise(real, rnd["pcm_eps"], th)
                with torch.no_grad():
                    xk = xh
                    for k in range(2):
                        tk = th - k * 10
                        xk = scheduler.jump(xk, guided(xk, tk, condition), tk, tk - 10)
                    target = scheduler.jump(xk, predict(student, xk, th - 20, condition), th - 20, pcm_boundary)
                boundary = scheduler.jump(xh, predict(student, xh, th, condition), th, pcm_boundary)
                pcm = (boundary - target).square().mean()
                (args.pcm_weight * pcm / plan.accumulation).backward()
                stats.append(torch.stack([dmd_value, gan_value, pcm.detach()]))
                del xh, xk, target, boundary, pcm, real, rnd, condition, dmd_value, gan_value
            if control("student", micro + 1):
                stopped = True
                break
        if stopped:
            break
        with measure("G_optimizer"):
            gnorm = clip_sharded(student)
            if args.offload_student_optimizer:
                for state in og.state.values():
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key] = state[key].to(device)
            og.step()
            og.zero_grad(set_to_none=True)
            if args.offload_student_optimizer:
                for state in og.state.values():
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key] = state[key].cpu()
        gcount += 1
        torch.cuda.synchronize()
        cycle_seconds = torch.tensor(time.monotonic() - cycle_start, device=device)
        dist.all_reduce(cycle_seconds, op=dist.ReduceOp.MAX)
        cycle_seconds = cycle_seconds.item()
        record({"phase": "student_done", "g": gcount, "stage": stage, "dmd": use_dmd, "gan": use_gan,
                "grad_norm": gnorm, "seconds": time.monotonic() - tick, "cycle_seconds": cycle_seconds,
                "student_samples_per_second": plan.global_batch / cycle_seconds,
                "original_indices_rank": originals, "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
                **summarize(stats, ["dmd_loss", "gan_loss", "pcm_loss"])})
        trace.step()
        if args.benchmark:
            benchmark_cycles.append(cycle_seconds)
        else:
            if args.save_every_updates and gcount % args.save_every_updates == 0:
                snapshot("periodic", "student_update_complete")
            if gcount % math.ceil(args.eval_every_samples / plan.global_batch) == 0:
                evaluate("periodic")
    trace.close()
    # All controlled exits are at a completed student-cycle boundary.
    og.zero_grad(set_to_none=True)
    of.zero_grad(set_to_none=True)
    if od is not None:
        od.zero_grad(set_to_none=True)
    if not args.benchmark and last_saved_gcount != gcount:
        snapshot("stopped" if stopped else "end", "finished")
    final_training_state = None
    if not args.benchmark:
        names = [stamp() if rank == 0 else None]
        dist.broadcast_object_list(names, src=0)
        final_training_state = str(out / "training_state" / f"{names[0]}_g{gcount:06d}")
        save_training(final_training_state, models, optimizers,
                      {"gcount": gcount, "fcount": fcount, "dcount": dcount}, config)
        print("TRAINING_STATE_SAVED", rank, final_training_state, flush=True)
    peak = torch.tensor([torch.cuda.max_memory_allocated() / 1024**3,
                         torch.cuda.max_memory_reserved() / 1024**3], device=device)
    dist.all_reduce(peak, op=dist.ReduceOp.MAX)
    torch.cuda.synchronize()
    if rank == 0:
        status = {"status": "stopped" if stopped else "complete", "timestamp": stamp(),
                  "student_updates": gcount, "fake_updates": fcount, "D_updates": dcount,
                  "effective_batch": plan.global_batch, "microbatch_per_gpu": plan.micro_batch,
                  "accumulation": plan.accumulation, "peak_allocated_GiB": peak[0].item(),
                  "peak_reserved_GiB": peak[1].item(),
                  "profile_rank0_seconds": dict(phase_times)}
        status["training_state"] = final_training_state
        if args.benchmark:
            measured = benchmark_cycles[args.benchmark_warmup:]
            status.update(benchmark=True, warmup_cycles=args.benchmark_warmup, measured_cycles=len(measured),
                          cycle_seconds=benchmark_cycles,
                          warmup_seconds=sum(benchmark_cycles[:args.benchmark_warmup]),
                          end_to_end_seconds=time.monotonic()-start,
                          actual_fake_updates_per_student=args.fake_updates,
                          trace_enabled=bool(args.trace_cycles),
                          student_samples_per_second=len(measured) * plan.global_batch / sum(measured) if measured else None,
                          note="Full training fake:G ratio and stage cycle; IO excluded, profile/trace runs not unbiased throughput")
            write_json(out / "benchmark.json", status)
        write_json(out / "status.json", status)
    log.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
