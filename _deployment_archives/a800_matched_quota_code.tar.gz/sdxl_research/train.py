"""Real batched SDXL TM/DMD/PixelGAN training; compatible with torch 2.5 FSDP2.

Example: torchrun --standalone --nproc_per_node=2 sdxl_research/train.py \
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

from batching import (BatchPlan, cfg_prediction, conditions,
                      r1_penalty, random_inputs, rollout_chunks, slice_batch)
from objectives import (VPSchedule, dmd_surrogate,
                        rollout_score_and_final)
from stage_policy import make_stage_plan
from gan_losses import discriminator_loss, generator_loss
from image_views import decode_view, augment_view
from pixel import PixelDiscriminator
from sam_pixel import SAMDiscriminator
import boundary as boundary_dm
from training_state import save_training, load_training, load_student_init
from runtime import average_gradients, TraceWindow, schedule_settings
from turning_sampling import TurningSampler


def shard_unet(model, mesh, policy, granularity='fine', offload=False):
    extra = {"offload_policy": CPUOffloadPolicy()} if offload else {}
    shard_types = (ResnetBlock2D, Transformer2DModel, Downsample2D, Upsample2D)
    if granularity == 'fine':
        shard_types += (BasicTransformerBlock,)
    for child in reversed(list(model.modules())):
        if isinstance(child, shard_types):
            fully_shard(child, mesh=mesh, mp_policy=policy, **extra)
    fully_shard(model, mesh=mesh, mp_policy=policy, **extra)
    return model


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
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("SDXL_ROOT", "/data5/shenglin.zeng/sdxl_crossdistill")))
    parser.add_argument("--run", required=True)
    parser.add_argument("--resume", help="Completed final training-state directory; same GPU count and settings")
    parser.add_argument("--init-student", type=Path, help="Student only from a completed state; fake/D/optimizers start fresh")
    parser.add_argument("--objective", choices=('clean','boundary'), default='clean')
    parser.add_argument("--boundary-noise-max", type=int, help="Optional shared upper re-noise limit; enables stage-conditioned fake")
    parser.add_argument("--boundary-margin", type=int, default=10)
    parser.add_argument("--fake-weight-max", type=float, default=10.)
    parser.add_argument("--grad-clip", type=float, help="Default 1 for boundary, 10 for legacy clean")
    parser.add_argument("--sam-path", type=Path, default=Path('models/sam-vit-base'))
    parser.add_argument("--gan-ramp-updates", type=int, default=0)
    parser.add_argument("--calibrate-only", type=int, default=0,
                        help="Boundary: probe N microbatches from a trained resume state; no optimizer updates")
    parser.add_argument("--resume-repack", action="store_true", help="Explicitly allow changing micro/rollout batch on resume; global batch and world size must stay fixed")
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/parallel_10k_seed20261018"),
                        help="Prepared dataset directory, absolute or relative to root")
    parser.add_argument("--variant", choices=("joint4", "gan4", "joint34", "dmd_only"), required=True)
    parser.add_argument("--student-stages", type=int, nargs='+', default=[2,3,4])
    parser.add_argument("--dmd-stages", type=int, nargs='*', help="Override variant DMD stages; empty disables DMD")
    parser.add_argument("--gan-stages", type=int, nargs='*', help="Override variant GAN stages; empty disables GAN")
    parser.add_argument("--fake-policy", choices=('crossdistill','student_cycle','match_student','final'), default='crossdistill')
    parser.add_argument("--gan-loss", choices=('hinge','logistic'), default='hinge')
    parser.add_argument("--discriminator", choices=('hybrid','rgb','sam'), default='hybrid')
    parser.add_argument("--gan-view", choices=('tile','full','whole'), default='tile', help="full=crop after full decode; whole=uncropped image")
    parser.add_argument("--gan-augment", choices=('d_only','both','none'), default='d_only')
    parser.add_argument("--schedule", type=int, nargs=4, default=[999,919,499,199])
    parser.add_argument("--pcm-boundary", type=int, help="Default: second sampling point")
    parser.add_argument("--dmd-t-max", type=int, default=919, help="DMD re-noise upper bound; independent of inference schedule")
    parser.add_argument("--match-fake-prior", action="store_true", help="Use the same stage-conditioned re-noise distribution for fake, with independent sampling")
    parser.add_argument("--turning-profile", type=Path, help="Optional student DMD noise prior; boundary objective conditions it on each valid stage range")
    parser.add_argument("--pcm-weight", type=float, default=10)
    parser.add_argument("--d-grad-bucket-mb", type=float, default=0, help="Opt-in bucket size; 0 keeps tested per-parameter reductions")
    parser.add_argument("--fake-adam-foreach", action="store_true", help="Opt-in: validate DTensor/device support before use")
    parser.add_argument("--student-adam-foreach", action="store_true")
    parser.add_argument("--compile-teacher", action="store_true", help="Opt-in frozen teacher only; include compile warmup in cost")
    parser.add_argument("--shard-teacher", action="store_true", help="FSDP-shard frozen teacher weights; no CPU offload")
    parser.add_argument("--control-mode", choices=("cycle", "legacy"), default="legacy")
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
    parser.add_argument("--checkpoint-mode", choices=("full","final-weights"), default="full",
                        help="final-weights: one inference-only student snapshot at exit, no periodic or optimizer states")
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
    if args.resume and args.init_student:
        parser.error('Choose resume OR student-only initialization')
    if args.calibrate_only < 0 or (args.calibrate_only and (not args.resume or args.objective!='boundary')):
        parser.error('Calibration requires a boundary resume state and positive microbatch count')
    if args.objective=='clean' and args.gan_ramp_updates:
        parser.error('GAN ramp is currently supported only by the boundary recipe')
    if args.grad_clip is None:
        args.grad_clip = 1. if args.objective=='boundary' else 10.
    if args.grad_clip <= 0 or args.fake_weight_max <= 0 or args.gan_weight < 0 or args.gan_ramp_updates < 0:
        parser.error('Invalid gradient/loss setting')
    if args.objective=='boundary':
        if args.variant not in ('joint4','dmd_only') or args.student_stages != [2,3,4] or args.dmd_stages is not None or args.gan_stages is not None or args.fake_policy!='crossdistill':
            parser.error('Boundary recipe has fixed middle+terminal routing; use joint4 or dmd_only without stage overrides')
        if args.variant=='joint4' and (args.discriminator,args.gan_view,args.gan_augment)!=('sam','whole','none'):
            parser.error('Boundary joint4 expects SAM, whole-image decode, no augmentation')
        try:
            boundary_dm.phase_ranges([*args.schedule,-1],args.boundary_margin,args.boundary_noise_max)
        except ValueError as error:
            parser.error(str(error))
        if args.pcm_boundary not in (None,args.schedule[1]):
            parser.error('Boundary PCM relay must equal the second sampling knot')
    if args.boundary_noise_max is not None and args.objective != "boundary":
        parser.error("--boundary-noise-max requires boundary objective")
    if args.match_fake_prior and (args.objective!="boundary" or not args.turning_profile):
        parser.error("--match-fake-prior requires boundary objective and a prior profile")
    if args.resume_repack and not args.resume:
        parser.error('--resume-repack requires --resume')
    if args.offload_student and args.offload_student_optimizer:
        parser.error("Use only one student offload mode; CPU-offloaded parameters already keep Adam state on CPU")
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
        make_stage_plan(args.variant,args.student_stages,args.dmd_stages,args.gan_stages,args.fake_policy)
    except ValueError as error:
        parser.error(str(error))
    if args.compile_teacher and os.environ.get("TORCHDYNAMO_DISABLE") == "1":
        parser.error("Unset TORCHDYNAMO_DISABLE to use --compile-teacher")
    return args


def main():
    args = arguments()
    if args.turning_profile and not args.turning_profile.is_absolute():
        args.turning_profile=args.root/args.turning_profile
    turning = TurningSampler(args.turning_profile,args.dmd_t_max) if args.turning_profile else None
    boundary_mode = args.objective == 'boundary'
    stage_plan = make_stage_plan(args.variant,args.student_stages,args.dmd_stages,args.gan_stages,args.fake_policy)
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    shared_yield_file = os.environ.get('SDXL_SHARED_YIELD_FILE')
    if shared_yield_file and (world!=1 or not args.benchmark):
        raise ValueError('Shared-GPU yielding is limited to single-rank short benchmarks')
    shared_yield_file = Path(shared_yield_file) if shared_yield_file else None
    shared_wait_seconds = 0.
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

    def load_model(train, offload=False, stage_condition=False):
        model = UNet2DConditionModel.from_pretrained(
            base / "unet", variant="fp16", torch_dtype=torch.float32 if train else torch.bfloat16,
            local_files_only=True).to(device)
        if stage_condition:
            boundary_dm.add_fake_stage_condition(model)
        model.requires_grad_(train).train(train)
        if train:
            model.enable_gradient_checkpointing()
            shard_unet(model, mesh, policy, args.shard_granularity, offload)
        return model

    print("LOAD", rank, "micro_batch", plan.micro_batch, "accumulation", plan.accumulation, flush=True)
    student = load_model(True, args.offload_student)
    if args.init_student:
        load_student_init(args.init_student, student, args.shard_granularity)
    fake = load_model(True, args.fake_offload, args.boundary_noise_max is not None)
    teacher = load_model(False)
    if args.shard_teacher:
        # Retain the original autocast inputs; only distribute frozen weights.
        teacher_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                              cast_forward_inputs=False)
        shard_unet(teacher, mesh, teacher_policy, args.shard_granularity)
    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", variant="fp16", torch_dtype=torch.float32,
                                      local_files_only=True).to(device).eval().requires_grad_(False)
    # Checkpoint individual decoder blocks as well as the outer tile call:
    # recomputing the whole decoder alone still retains all its inner activations.
    vae.enable_gradient_checkpointing()
    torch.manual_seed(args.seed + 7000)
    disc = None
    if stage_plan.gan:
        sam_path = args.sam_path if args.sam_path.is_absolute() else args.root / args.sam_path
        disc = (SAMDiscriminator(sam_path) if args.discriminator=='sam' else
                PixelDiscriminator(args.root / "models/dinov2-small", kind=args.discriminator)).to(device)
        disc.set_heads(False)
    adam = dict(betas=(0., .999), weight_decay=0.) if boundary_mode else dict(betas=(.9,.999),weight_decay=.01)
    og = torch.optim.AdamW(student.parameters(), lr=args.student_lr, **adam, foreach=args.student_adam_foreach)
    of = torch.optim.AdamW(fake.parameters(), lr=args.fake_lr, **adam, foreach=args.fake_adam_foreach)
    od = torch.optim.AdamW(disc.trainable(), lr=args.d_lr, betas=(0, .99), weight_decay=0, foreach=False) if disc is not None else None
    scheduler = VPSchedule(DDIMScheduler.from_pretrained(base, subfolder="scheduler", local_files_only=True).alphas_cumprod.to(device))
    knots, pcm_boundary, dmd_t_max = schedule_settings(args.schedule, args.dmd_t_max, args.pcm_boundary)
    ranges = boundary_dm.phase_ranges(knots,args.boundary_margin,args.boundary_noise_max) if boundary_mode else None
    latent_shape = tuple(data['latents'].shape[1:])
    if args.compile_teacher:
        teacher.forward = torch.compile(teacher.forward, mode="default", fullgraph=False)
    config = {**vars(args), "root": str(args.root), "data_dir": str(data_dir), "sam_path":str(args.sam_path),
              "init_student":str(args.init_student) if args.init_student else None, "world_size": world, "effective_batch": plan.global_batch,
              "accumulation": plan.accumulation, "microbatch_per_gpu": plan.micro_batch,
              "requested_training_batch": requested.global_batch, "schedule": knots,
              "pcm_boundary": pcm_boundary, "dmd_t_max": dmd_t_max,
              "pcm_t_range": [pcm_boundary + 21, 999], "student_dmd_t_range": [20,dmd_t_max],
              "runtime_versions": {"torch": torch.__version__, "cuda": torch.version.cuda,
                                   "device": torch.cuda.get_device_name(gpu)},
              "student_cfg": 1, "train_count": data["train_count"],
              "initialization": f"full training state from {args.resume}" if args.resume else "untouched SDXL Base; independent student/fake/D per run",
              "checkpoint_kind": ("one final BF16 student snapshot; inference only" if args.checkpoint_mode=="final-weights" else "periodic BF16 student weights; final FP32 student/fake/D + optimizers/counters/RNG"),
              "spectral_norm_policy": "20 initialization iterations; one refresh per D optimizer step, fixed across microbatches",
              "fake_rollout_batches_per_update_per_rank": math.ceil((plan.global_batch // world) / args.rollout_batch),
              "rollout_cache": "one local effective fake batch; GPU latents only plus its existing batch inputs; discarded before next student update",
              "batch_semantics": "mean loss / accumulation, followed by FSDP mean over ranks",
              "augmentation": "independent per-image crop/flip; sample RNG does not depend on batch packing",
              "gan_stages": list(stage_plan.gan), "dmd_stages": list(stage_plan.dmd),
              "fake_stage_policy": "crossdistill_cycle_1_2_3_4_v1" if args.fake_policy=='crossdistill' else args.fake_policy+'_v1',
              "experiment_modules": {**stage_plan.metadata(), "gan_loss":args.gan_loss,
                                     "discriminator":args.discriminator,"gan_view":args.gan_view,"gan_augment":args.gan_augment},
              "D_fake_stage": 4 if disc is not None else None,
              "discriminator_enabled": disc is not None,
              "benchmark_protocol": "same fake:G ratio and configured student stages as training; excludes evaluation/checkpoint IO; warmup recorded separately" if args.benchmark else None}
    if args.init_student:
        config['initialization'] = f'student only from {args.init_student}; fresh teacher-initialized fake and fresh D/optimizers'
    config['turning_profile'] = str(args.turning_profile) if args.turning_profile else None
    config['turning_sampling'] = dict(turning.metadata) if turning else None
    if turning and args.match_fake_prior:
        config['turning_sampling'].update(scope='student and fake re-noise; same per-stage distribution, independent rollouts and RNG; PCM unchanged', fake_seed_offset=500000)
    if boundary_mode:
        config.update(fake_stage_policy='boundary_2_3_4_v1', student_dmd_t_range=ranges,
                      D_updates_per_G=1 if disc is not None else 0,
                      latent_shape=latent_shape, optimizer_settings=adam,
                      spectral_norm_policy='not used by SAM heads',
                      augmentation='whole-image SAM; no flips or latent crops')
        config['experiment_modules'] = dict(objective='boundary_v1', margin=args.boundary_margin,
            ranges={str(k):list(v) for k,v in ranges.items()}, fake_weight_max=args.fake_weight_max, grad_clip=args.grad_clip,
            student='random middle 2/3 plus terminal4 each G', fake='cycle 2/3/4',
            terminal_dmd=True, gan=bool(disc is not None), discriminator=args.discriminator,
            gan_view=args.gan_view, gan_loss=args.gan_loss, gan_augment=args.gan_augment,
            gan_ramp_updates=args.gan_ramp_updates, sam_path=str(args.sam_path),
            fake_normalization='conditional epsilon; clipped inverse variance',
            student_normalization='boundary posterior mean; 0.5 MSE')
    if rank == 0:
        write_json(out / "config.json", config)
        write_json(out / "data_manifest.json", manifest)
        source = out / "source"
        source.mkdir(exist_ok=True)
        hashes = {}
        for name in ("train.py", "batching.py", "pixel.py", "objectives.py", "training_state.py", "runtime.py",
                     "stage_policy.py", "gan_losses.py", "image_views.py", "boundary.py", "sam_pixel.py", "turning_sampling.py"):
            file = Path(__file__).with_name(name)
            shutil.copyfile(file, source / name)
            hashes[name] = hashlib.sha256(file.read_bytes()).hexdigest()
        if turning:
            shutil.copyfile(turning.path,source/'turning_distribution.json')
            hashes['turning_distribution.json']=turning.metadata['distribution_sha256']
        write_json(out / "source_sha256.json", hashes)
    log = (out / f"rank{rank}.jsonl").open("w", encoding="utf-8")
    gcount = fcount = dcount = 0
    models = {"student": student, "fake": fake, "discriminator": disc}
    optimizers = {"og": og, "of": of, "od": od}
    if args.resume:
        progress = load_training(args.resume, models, optimizers, config, allow_repacking=args.resume_repack)
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
    benchmark_wait_cycles = []
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

    def predict(model, x, t, condition, negative=False, score_stage=None):
        nonlocal shared_wait_seconds
        if shared_yield_file and shared_yield_file.exists():
            wait_start=time.monotonic()
            while shared_yield_file.exists():time.sleep(.5)
            shared_wait_seconds+=time.monotonic()-wait_start
        with torch.autocast("cuda", dtype=torch.bfloat16):
            extra = {}
            if model is fake and args.boundary_noise_max is not None:
                extra["class_labels"] = boundary_dm.stage_labels(score_stage, x)
            return model(x, t, **extra,
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

    def training_batch(update, micro, phase, score_stage=None):
        positions = plan.positions(update, micro, rank)
        indices = [order[pos % len(order)] for pos in positions]
        condition = conditions(data, indices, device)
        real = data["latents"].index_select(0, torch.tensor(indices)).to(device, dtype=torch.float32)
        randoms = random_inputs(positions, phase, device, seed=args.seed,
                               pcm_boundary=pcm_boundary, dmd_t_max=dmd_t_max, shape=latent_shape,
                               time_range=ranges[score_stage] if boundary_mode else None)
        if turning and phase=='student' and not boundary_mode:
            turning.apply(randoms,positions,phase,args.seed)
            record({'phase':'turning_sample','g':update,'micro':micro,'timesteps':randoms['t'].tolist()})
        return indices, condition, real, randoms

    def generate_fake_batch(update):
        # Student weights are fixed for the entire fake/D update. Generate its
        # samples with larger no-grad batches, then slice this cache into the
        # unchanged backward microbatches. No samples are reused across updates.
        positions = plan.rank_positions(update, rank)
        indices = [order[pos % len(order)] for pos in positions]
        condition = conditions(data, indices, device)
        score_stage = boundary_dm.critic_stage(update) if boundary_mode else stage_plan.critic_stage(update, gcount)
        randoms = random_inputs(positions, "fake", device, seed=args.seed, shape=latent_shape,
                               time_range=ranges[score_stage] if boundary_mode else None)
        if boundary_mode:
            if turning and args.match_fake_prior:
                turning.apply(randoms,positions,'fake',args.seed+500000+1000*score_stage,
                              time_range=ranges[score_stage],include_fake=True)
                record({'phase':'fake_turning_sample','fake_update':update,'stage':score_stage,
                        'timesteps':randoms['t'].tolist()})
            generated, final = rollout_chunks(lambda c,n: boundary_dm.rollout_boundary_and_final(
                lambda x,t,c:predict(student,x,t,c),scheduler,c,n,knots,score_stage),
                condition,randoms['noise'],args.rollout_batch)
        elif disc is None:
            generated = rollout_chunks(lambda c, n: rollout(c, n, score_stage),
                                       condition, randoms["noise"], args.rollout_batch)
            final = None
        else:
            def paired(c, n):
                return rollout_score_and_final(lambda x, t, c: predict(student, x, t, c),
                                               scheduler, c, n, knots, score_stage)
            generated, final = rollout_chunks(paired, condition, randoms["noise"], args.rollout_batch)
        # Initial rollout noise is not needed once all final latents are cached.
        del randoms["noise"]
        real = data["latents"].index_select(0, torch.tensor(indices)).to(device, dtype=torch.float32)
        return indices, condition, real, randoms, generated, final

    def clip_sharded(model, clip=True):
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
        if clip:
            torch._foreach_mul_(gradients, min(1., args.grad_clip / (norm + 1e-12)))
        return norm

    def pcm_loss(real, rnd, condition):
        th = rnd['pcm_t']
        xh = scheduler.noise(real,rnd['pcm_eps'],th)
        with torch.no_grad():
            xk = xh
            for k in range(2):
                tk = th-k*10
                xk = scheduler.jump(xk,guided(xk,tk,condition),tk,tk-10)
            target = scheduler.jump(xk,predict(student,xk,th-20,condition),th-20,pcm_boundary)
        prediction = scheduler.jump(xh,predict(student,xh,th,condition),th,pcm_boundary)
        return (prediction-target).square().mean()

    def boundary_loss(generated, k, rnd, condition):
        with torch.no_grad():
            xt = boundary_dm.renoise(scheduler,generated.detach(),knots[k],rnd['t'],rnd['eps'])
            ef = predict(fake,xt,rnd['t'],condition,score_stage=k)
            et = guided(xt,rnd['t'],condition)
        return boundary_dm.dm_surrogate(scheduler,generated,knots[k],rnd['t'],xt,ef,et)[0]

    def phase_random(update, micro, k):
        positions=plan.positions(update,micro,rank)
        rnd=random_inputs(positions,'student',device,
            seed=args.seed+1000*k,shape=latent_shape,pcm_boundary=pcm_boundary,time_range=ranges[k])
        if turning:
            turning.apply(rnd,positions,'student',args.seed+1000*k,time_range=ranges[k])
            record({'phase':'boundary_turning_sample','g':update,'micro':micro,'stage':k,'timesteps':rnd['t'].tolist()})
        return rnd

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
            flags[0] = int(not args.benchmark and args.checkpoint_mode=="full" and not args.save_every_updates and now - last_snapshot >= args.snapshot_seconds)
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
            noise = torch.stack([torch.randn(latent_shape, device=device,
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
                    tile = decode_view(vae, pair, [(40, 40)] * 2, kind=args.gan_view)
                    parts = disc(tile, c["pooled_prompt_embeds"][j:j + 1].expand(2, -1), return_parts=True)
                    keys = ('total','features') if args.discriminator=='sam' else ('total','dino','rgb')
                    row["D_real_fake"] = {key: parts[key].tolist() for key in keys}
                rows.append(row)
        write_json(dest / f"rank{rank}.json", {"student_updates": gcount,
                   "student_samples": gcount * plan.global_batch, "samples": rows})
        dist.barrier()
        print("EVALUATED", rank, gcount, flush=True)

    if args.calibrate_only:
        if disc is None or dcount == 0:
            raise ValueError('GAN calibration needs an already trained discriminator')
        probes = []
        for i in range(args.calibrate_only):
            update = gcount+i
            mid = boundary_dm.middle_stage(update,args.seed)
            _,condition,real,rnd = training_batch(update,0,'student',mid)
            og.zero_grad(set_to_none=True)
            for k,y in boundary_dm.local_rollout(lambda x,t,c:predict(student,x,t,c),
                    scheduler,condition,rnd['noise'],knots,mid):
                boundary_loss(y,k,phase_random(update,0,k),condition).backward()
            dm_norm = clip_sharded(student,clip=False)
            og.zero_grad(set_to_none=True)
            pcm_loss(real,rnd,condition).backward()
            pcm_norm = clip_sharded(student,clip=False)
            og.zero_grad(set_to_none=True)
            y=rollout(condition,rnd['noise'],4,True)
            terminal=boundary_loss(y,4,phase_random(update,0,4),condition)
            terminal.backward()
            terminal_norm=clip_sharded(student,clip=False)
            og.zero_grad(set_to_none=True)
            # Recompute final call, not retain multiple full-network graphs.
            y=rollout(condition,rnd['noise'],4,True)
            rgb=decode_view(vae,y,rnd['crops'],kind='whole',grad=True,chunk_size=args.vae_batch)
            generator_loss(disc(rgb,condition['pooled_prompt_embeds']),args.gan_loss).backward()
            gan_norm=clip_sharded(student,clip=False)
            og.zero_grad(set_to_none=True)
            if min(dm_norm,pcm_norm,terminal_norm,gan_norm) < 1e-10:
                raise ValueError('Near-zero calibration gradient; inspect critic and samples before selecting weights')
            probes.append(dict(dm=dm_norm,pcm=pcm_norm,terminal=terminal_norm,gan=gan_norm))
            del y,rgb,terminal,real,rnd,condition
        if rank==0:
            import statistics
            write_json(out/'calibration.json',dict(source=args.resume,G_updates=0,D_updates=0,
                batch_per_probe=plan.micro_batch*world,probes=probes,
                suggested_pcm_weight=statistics.median(p['dm']/p['pcm'] for p in probes),
                suggested_gan_weight=.2*statistics.median(p['terminal']/p['gan'] for p in probes),
                note='Parameter-gradient ratios; suggestions only, not applied automatically'))
        log.close()
        dist.barrier()
        dist.destroy_process_group()
        return

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
        stage = boundary_dm.middle_stage(gcount,args.seed) if boundary_mode else stage_plan.student_stage(gcount)
        repeats = args.fake_updates
        for fake_repeat in range(repeats):
            tick = time.monotonic()
            of.zero_grad(set_to_none=True)
            train_d = disc is not None and (not boundary_mode or fake_repeat==repeats-1)
            if train_d:
                od.zero_grad(set_to_none=True)
                disc.set_heads(True)
            stats, originals = [], []
            with measure("fake_student_rollout"):
                cached_indices, cached_condition, cached_real, cached_randoms, cached_generated, cached_final = generate_fake_batch(fcount)
            for micro in range(plan.accumulation):
                begin, end = micro * plan.micro_batch, (micro + 1) * plan.micro_batch
                indices = cached_indices[begin:end]
                condition = slice_batch(cached_condition, begin, end)
                real = cached_real[begin:end]
                rnd = slice_batch(cached_randoms, begin, end)
                generated = cached_generated[begin:end]
                originals.extend(data["original_indices"][i] for i in indices)
                dvalues = torch.empty(0, device=device)
                if train_d:
                    with measure("D_decode"), torch.no_grad():
                        real_rgb = augment_view(decode_view(vae, real, rnd["crops"], kind=args.gan_view), rnd["flips"], 'D', args.gan_augment)
                        fake_rgb = augment_view(decode_view(vae, cached_final[begin:end], rnd["crops"], kind=args.gan_view), rnd["flips"], 'D', args.gan_augment)
                    with measure("D_forward_backward"):
                        r1_on = args.discriminator=='sam' or dcount % 16 == 15
                        real_rgb.requires_grad_(r1_on and args.discriminator!='sam')
                        extra = {'feature_r1':True} if args.discriminator=='sam' else {}
                        pr = disc(real_rgb, condition["pooled_prompt_embeds"], return_parts=True, **extra)
                        pf = disc(fake_rgb, condition["pooled_prompt_embeds"], return_parts=True)
                        hinge = discriminator_loss(pr["total"], pf["total"], args.gan_loss)
                        r1 = pr['r1'] if args.discriminator=='sam' else (r1_penalty(pr["total"], real_rgb) if r1_on else torch.zeros((), device=device))
                        dloss = hinge + (1e-5 if args.discriminator=='sam' else .05*16) * r1
                        feature_key = 'features' if args.discriminator=='sam' else 'dino'
                        dvalues = torch.stack([hinge, r1, pr["total"].mean(), pf["total"].mean(),
                                              pr[feature_key].mean(), pf[feature_key].mean(), pr["rgb"].mean(), pf["rgb"].mean()]).detach()
                        (dloss / plan.accumulation).backward()
                        del real_rgb, fake_rgb, pr, pf, hinge, r1, dloss
                with measure("fake_forward_backward"):
                    if boundary_mode:
                        s = knots[boundary_dm.critic_stage(fcount)]
                        xt = boundary_dm.renoise(scheduler,generated.detach(),s,rnd['t'],rnd['eps'])
                        floss = boundary_dm.fake_loss(scheduler,generated,s,rnd['t'],rnd['eps'],
                                  predict(fake,xt,rnd['t'],condition,score_stage=boundary_dm.critic_stage(fcount)),args.fake_weight_max)
                    else:
                        xt = scheduler.noise(generated, rnd["eps"], rnd["t"])
                        floss = (predict(fake, xt, rnd["t"], condition) - rnd["eps"]).square().mean()
                    (floss / plan.accumulation).backward()
                    stats.append(torch.cat([floss.detach().reshape(1), dvalues]))
                    del xt, floss, generated, real, rnd, condition, dvalues
                if control("fake_D" if disc is not None else "fake", micro + 1):
                    stopped = True
                    break
            del cached_indices, cached_condition, cached_real, cached_randoms, cached_generated, cached_final
            if stopped:
                break
            if train_d:
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
            dcount += int(train_d)
            torch.cuda.synchronize()
            feature_name = 'sam' if args.discriminator=='sam' else 'dino'
            dmetrics = ["D_hinge" if args.gan_loss=='hinge' else "D_logistic", "R1_raw", "D_real", "D_fake", f"D_{feature_name}_real", f"D_{feature_name}_fake", "D_rgb_real", "D_rgb_fake"] if train_d else []
            record({"phase": "fake_D" if train_d else "fake", "fake_update": fcount, "D_update": dcount,
                    "fake_stage": boundary_dm.critic_stage(fcount-1) if boundary_mode else stage_plan.critic_stage(fcount - 1, gcount), "D_fake_stage": 4 if train_d else None,
                    "seconds": time.monotonic() - tick, "fake_grad_norm": fnorm,
                    "rollout_batch": args.rollout_batch,
                    "rollout_batches_per_rank": math.ceil((plan.global_batch // world) / args.rollout_batch),
                    "original_indices_rank": originals,
                    **({"D_grad_norm": dnorm, "R1_active": r1_on, "D_collectives":d_collectives} if train_d else {}),
                    **summarize(stats, ["fake_loss"] + dmetrics)})
        if stopped:
            break
        og.zero_grad(set_to_none=True)
        tick = time.monotonic()
        stats, originals, boundary_stats = [], [], []
        use_dmd = True if boundary_mode else stage in stage_plan.dmd
        use_gan = disc is not None if boundary_mode else stage in stage_plan.gan
        for micro in range(plan.accumulation):
            indices, condition, real, rnd = training_batch(gcount, micro, "student", stage)
            originals.extend(data["original_indices"][i] for i in indices)
            if boundary_mode:
                dmd_value = gan_value = torch.zeros((), device=device)
                middle_value = terminal_value = torch.zeros((), device=device)
                for k, generated in boundary_dm.local_rollout(
                        lambda x,t,c:predict(student,x,t,c), scheduler, condition, rnd["noise"], knots, stage):
                    # Separate deterministic RNG stream per phase; packing independent.
                    phase_rnd = phase_random(gcount,micro,k)
                    with measure("G_boundary_objective_backward"):
                        dmd = boundary_loss(generated,k,phase_rnd,condition)
                        loss = dmd
                        dmd_value = dmd_value + dmd.detach()
                        if k == 4:
                            terminal_value = dmd.detach()
                            if disc is not None:
                                rgb = decode_view(vae,generated,rnd['crops'],kind=args.gan_view,
                                                  grad=True,chunk_size=args.vae_batch)
                                adversarial = generator_loss(disc(rgb,condition['pooled_prompt_embeds']),args.gan_loss)
                                if args.gradient_probe_every and gcount % args.gradient_probe_every==0 and micro==0:
                                    gg=torch.autograd.grad(adversarial,generated,retain_graph=True)[0]
                                    gd=torch.autograd.grad(dmd,generated,retain_graph=True)[0]
                                    record({'phase':'boundary_GAN_gradient_probe','g':gcount,
                                        'GAN_latent_rms':(gg.square().mean().sqrt()*plan.micro_batch).item(),
                                        'terminal_DM_latent_rms':(gd.square().mean().sqrt()*plan.micro_batch).item()})
                                    del gg,gd
                                ramp = min(1.,(gcount+1)/args.gan_ramp_updates) if args.gan_ramp_updates else 1.
                                loss = loss + args.gan_weight*ramp*adversarial
                                gan_value = adversarial.detach()
                        else:
                            middle_value = dmd.detach()
                        (loss / plan.accumulation).backward()
                        del dmd, loss, phase_rnd
                        if k==4 and disc is not None:
                            del rgb, adversarial
                boundary_stats.append(torch.stack([middle_value,terminal_value]))
                del generated
            else:
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
                        rgb = decode_view(vae, generated, rnd["crops"], kind=args.gan_view, grad=True, chunk_size=args.vae_batch)
                        rgb = augment_view(rgb, rnd["flips"], 'G', args.gan_augment)
                        adversarial = generator_loss(disc(rgb, condition["pooled_prompt_embeds"]), args.gan_loss)
                        if args.gradient_probe_every and stage_plan.gan_updates_before(gcount) % args.gradient_probe_every == 0 and micro == 0:
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
                pcm = pcm_loss(real,rnd,condition)
                (args.pcm_weight * pcm / plan.accumulation).backward()
                stats.append(torch.stack([dmd_value, gan_value, pcm.detach()]))
                del pcm, real, rnd, condition, dmd_value, gan_value
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
                **({'supervised_stages':[stage,4], 'GAN_stage':4 if use_gan else None} if boundary_mode else {}),
                "grad_norm": gnorm, "seconds": time.monotonic() - tick, "cycle_seconds": cycle_seconds,
                "student_samples_per_second": plan.global_batch / cycle_seconds,
                "original_indices_rank": originals, "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                "peak_reserved_GiB": torch.cuda.max_memory_reserved() / 1024**3,
                **summarize(stats, ["dmd_loss", "gan_loss", "pcm_loss"]),
                **(summarize(boundary_stats,["middle_dm_loss","terminal_dm_loss"]) if boundary_mode else {})})
        trace.step()
        if args.benchmark:
            benchmark_cycles.append(cycle_seconds)
            benchmark_wait_cycles.append(shared_wait_seconds-sum(benchmark_wait_cycles))
        else:
            if args.checkpoint_mode=="full" and args.save_every_updates and gcount % args.save_every_updates == 0:
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
    if not args.benchmark and args.checkpoint_mode=="full":
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
                          cycle_wait_seconds=benchmark_wait_cycles,
                          shared_wait_seconds=shared_wait_seconds,
                          warmup_seconds=sum(benchmark_cycles[:args.benchmark_warmup]),
                          training_phase_seconds=time.monotonic()-start,
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
