SDXL research execution and controlled experiments — 2026-10-22

Status
------
This is an isolated copy of the existing SDXL trainer. It does not replace
sdxl_parallel or change historical runs. CPU/Gloo checks have passed; full SDXL
GPU speed, memory, compile, and block-FSDP equivalence remain to be measured.
No long training or automatic monitoring/queue is started by these files.

Training logic
--------------
One shared student UNet, one fake-score UNet, one frozen teacher, and frozen
VAE/DINO with trainable RGB/DINO discriminator heads. Each G update follows five
fake/D updates. G stages cycle 2,3,4; high-noise PCM is applied on every G update.
joint4 adds GAN only at stage4; prefixes are detached. All parameters in the shared
student can update: "step4" does NOT mean training only one network layer.

The main uniform configs use inference knots 999/749/499/249, PCM boundary749,
and retain the original DMD re-noise range20..919. The additional
uniform_aligned_dmd749 config moves that range to20..749 for a separate ablation.
sampler_only_joint4 keeps PCM boundary919 too; it is an intentionally mismatched
diagnostic control, not the recommended ordinary four-stage training recipe.

Environment and assets
----------------------
Use the saved environment.json versions as the tested CPU reference (Python3.12,
torch2.5.1+cu121). On the 4090 server, create a separate environment rather than
modifying the existing personal sherly environment (torch2.10, no diffusers).
The new target root may be /data1/shenglin.zeng/sdxl_crossdistill. It must contain:
  models/sdxl-base-1.0/       original fp16 UNet/VAE/text encoder files and configs
  models/dinov2-small/
  datasets/parallel_10k_seed20261018/{prepared.pt,manifest.json}
For evaluation also copy datasets/dmd2/coco10k.zip and model/metric assets used
by the existing evaluator. Compare SHA256 after transfer. No asset transfer or
4090 environment installation was performed in this optimization task.

CPU validation (CUDA disabled)
------------------------------
cd sdxl_research
CUDA_VISIBLE_DEVICES='' python -m unittest -v test_runtime test_batching test_experiments
CUDA_VISIBLE_DEVICES='' torchrun --standalone --nproc_per_node=2 test_distributed_cpu.py
The first tests expect the sibling sdxl_parallel/batching.py baseline for input
equivalence; include that baseline file when validating a standalone deployment.

Preview and short GPU test
-------------------------
ROOT=/data1/shenglin.zeng/sdxl_crossdistill
python sdxl_research/launch.py --root "$ROOT" --gpus 2,7 \
  --config sdxl_research/configs/uniform_joint4.json --short

This prints a command only. Add --execute once those exact GPUs are confirmed
available. The launcher checks reset/N/A states, compute processes and free memory.
It does not reset/kill/steal GPUs. This check is not a reservation: coordinate
ownership for a long experiment. It runs in the foreground and writes logs.
For four approved cards, pass their four indices; batch64/micro16 implies one
accumulation pass on four cards, two on two cards.

The short test runs FOUR full G updates with five fake/D updates each, including
stage4 GAN, R1 at D16, evaluation, periodic weight save and final full state.
Verify all four ranks exited successfully, finite losses/gradients, >4GiB device
headroom at peak, and generated images before a fresh full run. Do NOT resume
the short-test weights into the scientific experiment.

Performance comparison
----------------------
python sdxl_research/benchmark_suite.py --root "$ROOT" --gpus 2,7 \
  --candidates legacy,bucket_cycle,block
Add --execute for real measurements. Each candidate uses9 G updates, fake:G5:1,
warmup3, measured6, and all three stages. No eval/checkpoint IO is timed.
Repeat the best pair in ABBA order under the same GPU load, then use18 cycles
for confirmation. Foreach/teacher_cfg/compile_teacher are independent opt-ins.
The old 1:1 benchmark is NOT a valid formal-speed comparator.
configs/performance/trace.json produces a rank0 Chrome trace; exclude its time
from speed comparisons. Compile warmup is reported separately. No fused AdamW,
no RNG vectorization, no cached stale teacher/fake outputs, no change of5:1 ratio.

python sdxl_research/summarize.py "$ROOT/outputs/perf/RUN_A" "$ROOT/outputs/perf/RUN_B" --out speed.json

Formal experiment (only after short-test review)
-----------------------------------------------
python sdxl_research/launch.py --root "$ROOT" --gpus 2,7 \
  --config sdxl_research/configs/uniform_joint4.json --execute
Defaults: fresh SDXL, G160, batch64, LR1e-6, save G40/80/120/160, evaluate every
two G updates, final FP32 models/Adam/counters/RNG. One final full state only.
To resume, put its complete training_state directory in a config's "resume"
field and choose a fresh --run. Keep world size, microbatch, FSDP layout,
optimizer options, data path and scientific settings identical.

Primary scientific matrix
-------------------------
legacy_dmd / legacy_joint4 / uniform_dmd / uniform_joint4.
All four use same seed/data/batch/lr/DMD time support. The schedule/PCM partition
factor and GAN factor are separated. Pick one validated runtime preset and apply
it to EVERY arm, including re-running the old-schedule reference. Historical
results are context; do not mix old-runtime/old-world-size runs as the only control.
Optional followups only after a positive result: uniform_gan4, uniform_joint34,
uniform_joint4_gan_half/double, uniform_aligned_dmd749, sampler_only_joint4.

Evaluation
----------
Copy configs/models.example.json and replace its checkpoint path explicitly.
python sdxl_research/evaluate.py prepare --root "$ROOT" --out "$ROOT/outputs/eval_new" --models models.json --n 10000
CUDA_VISIBLE_DEVICES=2 python sdxl_research/evaluate.py worker --root "$ROOT" --out "$ROOT/outputs/eval_new" --rank 0 --world 2
CUDA_VISIBLE_DEVICES=7 python sdxl_research/evaluate.py worker --root "$ROOT" --out "$ROOT/outputs/eval_new" --rank 1 --world 2
python sdxl_research/evaluate.py metrics --root "$ROOT" --out "$ROOT/outputs/eval_new"
python sdxl_research/feature_metrics.py --real "$ROOT/outputs/eval_new/features/real" --generated "$ROOT/outputs/eval_new/features/uniform_joint4" --out kid.json

Workers are independent, not torchrun; wait for both before metrics. The sampler
reads each saved model's schedule. FID/Patch-FID transforms remain identical to
the existing DMD2-based evaluator. Features are saved for every image; only128
preselected images per model are kept. --n1000 is development-only; never compare
its absolute FID with10K scores or claim it is a sealed test after tuning on it.
Patch-FID uses one center crop, so add blinded full-image/texture inspection and
semantic/diversity checks before interpreting an improvement as texture quality.

Budget
------
python sdxl_research/budget.py --seconds-per-update 119 --hours 72 --runs 2 --overhead-hours 12
Replace119 with a measurement on the actual available GPUs. Count wall-clock,
GPU-hours, full5:1 work, evaluation and checkpoint time. G160 gives only53 stage4
GAN updates and800 D updates; it is a pilot, not evidence of convergence.

Report
------
../output/sdxl-crossdistill/optimization_plan_20261022/report.html
SD3-Medium is a separate planned migration, not implemented by this SDXL trainer.
