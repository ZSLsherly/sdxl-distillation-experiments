SDXL true-batch training
========================

dmd_only control (2026-10-18): keep high-noise TM/PCM and stage2/3/4 DMD.
Do not instantiate the discriminator, train its heads, compute R1, or use GAN
in any student stage. The fake score network still updates 5 times per student
update. Dataset, per-sample RNG, knots, teacher CFG and learning rates remain
unchanged. start_dmd_only.sh starts a fresh 4-GPU batch64 run from SDXL Base.
Evaluate every 2 updates; save student weights about every 30 minutes and at
the end. The lower checkpoint frequency avoids filling the remaining disk.

joint34 (2026-10-18): DMD remains on stages 2/3/4; GAN is applied to the
predicted clean image at stages 3 and 4, with the same coefficient per stage.
Earlier rollout steps remain detached. The fake network and discriminator
continue using four-step student samples, as in joint4; their update rule,
dataset, seeds, learning rates, R1 and high-noise PCM are unchanged.
Four GPUs with micro-batch16 give global batch64 with accumulation1.
The joint34 short probe exercises stages 2/3/4 and all optimizer states.
The formal launcher is start_joint34.sh, gated on this probe's success.
Probe weights are discarded; formal training starts from original SDXL Base.

This is a standalone source directory. The old v4 experiment was stopped at
the user's request before deployment. Use the existing server environment.

Actual training batch:
  effective batch = GPU count * micro-batch per GPU * accumulation
  --batch 64 --micro-batch 2 on 2 GPUs -> 16 accumulation passes
  --batch 64 --micro-batch 4 on 2 GPUs -> 8 accumulation passes
  --batch 64 --micro-batch 8 on 2 GPUs -> 4 accumulation passes

UNet, fake score, VAE training tiles and DINO/RGB discriminator receive true
multi-image batches. Only per-image RNG/crop gathering and full-frame VAE image
export use small loops. Each sample keeps its own prompt, crop and noise levels.

Independent sample-generation batching (enabled by default):
  --micro-batch 2 --rollout-batch 8
Each fake update generates its entire local effective batch before backward
passes. For batch64 on 2 GPUs this means 32 samples per GPU: 4 rollout batches
of 8, followed by 16 training microbatches of 2. The original sample/noise stream
is preserved, and fake/D still take exactly 5 optimizer updates per G update.
The generated cache is discarded after each fake/D update; it is never reused
after student weights change. Only final latent outputs are cached, without
activation graphs. The default cache contains 32 FP32 latents, approximately
8 MiB per GPU, in addition to that batch's conditions/real latents/noise tensors.
--rollout-batch 2 provides the training-batch-sized baseline for comparison.

CPU validation (no downloads, no GPUs):
  PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' python sdxl_parallel/test_batching.py

On an AVAILABLE GPU pair, compare physical batches with a short benchmark:
  CUDA_VISIBLE_DEVICES=0,1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
    torchrun --standalone --nproc_per_node=2 sdxl_parallel/train.py \
    --run parallel_probe/joint4_mb2_UNIQUE --variant joint4 \
    --batch 64 --micro-batch 2 --rollout-batch 8 --benchmark --profile

Repeat with micro-batch 1, 4 or 8 and fresh run names. Benchmark uses a FIXED
global --benchmark-batch 16 by default, one fake/D update and one stage-4 G
update per cycle, for 3 cycles. This keeps sample work equal across candidates
and actually exercises rollout batch8. Use --benchmark-batch 64 for a longer
memory/timing probe of the formal effective batch.
The first cycle initializes optimizer states and exercises R1; the last two
measure steady throughput. It does NOT claim to validate the full 5:1 cycle or
batch64 training convergence. Compare benchmark.json student_samples_per_second
and peak_allocated_GiB. An OOM terminates the probe: reduce micro-batch and rerun
in a fresh process. No catch/retry inside an FSDP backward or partial optimizer
update. Choose the faster batch that fits, rather than simply the largest one.

Formal independent pair, after selecting an available set of GPUs:
  MICRO_BATCH=2 ROLLOUT_BATCH=8 bash sdxl_parallel/run_pair.sh

The pair launcher runs in the foreground; use a persistent terminal if needed.
It does not stop old runs or automatically occupy additional GPUs for tuning.
Run names are timestamped. Both variants use identical data/initialization,
batch size and random sample streams. Both start from original SDXL Base.
Default learning rates: student/fake 1e-6, D heads 1e-4; teacher CFG8;
5 fake/D updates per G; knots 999/919/499/199/clean; high-noise PCM weight10.
Only stage4 DMD is omitted in gan4. All timesteps share the same student UNet.

The pair launcher uses a fresh seeded selection of 10,000 training images and
32 held-out images from official DMD2 LAION shards 000/001. The prepared split
is datasets/parallel_10k_seed20261018; manifest.json records source rows.
Save student BF16 shards approximately every 30 minutes and at the end, with timestamps.
Earlier runs saved every 2 updates; their redundant checkpoints were pruned
on 2026-10-18, retaining final and selected comparison nodes. All evaluation
images and logs remain available; outputs/reviews/storage_cleanup_20261018.json
records the exact retained and deleted checkpoint directories.
No fake/D/Adam snapshots and no exact training resume.
Evaluate every 128 G samples (2 G updates at batch64), export train/holdout
images and D diagnostics. First stage4 GAN update is G3; evaluation at G4 is
the first regular comparison that includes the objective difference.

The GAN gradient diagnostic is OFF by default (no extra backward each batch).
--gradient-probe-every N enables an occasional first-microbatch probe.
--profile adds synchronized timings for rollout, decoding, D, fake, PCM and
optimizers; leave it off for normal throughput. Every cycle always records
wall time and effective student samples per second.

Memory: fake CPU offload remains enabled by default. --no-fake-offload is an
optional benchmark setting and is not assumed to fit two 40GB A100s. FSDP
reduces gradients on each microbatch to preserve sharded gradient memory.
Larger physical batches amortize communication without unsharded no_sync grads.

Optional benchmark switches (not enabled automatically):
  --batched-teacher-cfg combines conditional/unconditional teacher evaluation
    into one 2*micro-batch forward. It uses the same guidance formula but can
    increase activation memory. Default is separate teacher forwards.
  --shard-granularity block wraps transformer blocks without also wrapping
    each inner BasicTransformerBlock, reducing collective count. Peak memory
    can increase; the default fine layout preserves the previous topology.
Only enable switches that improve measured throughput and fit in GPU memory.
If a rollout probe runs out of memory, reduce --rollout-batch; if a backward
probe runs out of memory, reduce --micro-batch. Global training batch stays64.

The discriminator retains the existing DINOv2-S/14 + RGB design; this is not
an exact reproduction of SenseFlow/Flash-DMD. Tiled VAE decoding preserves
the previous approximation and is not identical to full-frame decoding.
GAN VAE decoding uses at most 4 tiles per call (--vae-batch), independently
of the physical UNet/discriminator batch. It preserves per-image gradients.
VAE decoder blocks use activation checkpointing; checkpointing only the outer
decode function was insufficient for the 8-image GAN backward on 40GB cards.
Spectral-normalization vectors are refreshed once per D optimizer update and
held fixed across its microbatches; their iteration count does not grow when
microbatch size decreases.
The pair launcher enables --offload-student-optimizer: student Adam moments
stay on CPU between updates and are copied to GPU only for Adam.step().
Student parameters stay on GPU throughout. This frees about 9.6 GiB per GPU
during forward/backward without offloading every student layer invocation.
