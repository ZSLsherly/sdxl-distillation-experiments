set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
cd /data5/shenglin.zeng/sdxl_crossdistill
mkdir -p sdxl_research
tar -xzf sdxl_research_20261022.tar.gz -C sdxl_research
cd sdxl_research
python -m unittest -v test_runtime test_batching test_experiments 2>&1 | tee cpu_tests.log
torchrun --standalone --nproc_per_node=2 test_distributed_cpu.py 2>&1 | tee distributed_cpu.log
python train.py --help > cli_help.txt
python evaluate.py --help > evaluation_help.txt
python launch.py --root /data5/shenglin.zeng/sdxl_crossdistill --gpus 2,7 --config configs/uniform_joint4.json --short > launch_dry_run.txt
python audit_data.py --prepared ../datasets/parallel_10k_seed20261018/prepared.pt --coco-zip ../datasets/dmd2/coco10k.zip --out data_audit.json
python - <<'PY' > environment.json
import importlib.metadata,json,sys
names=['torch','torchvision','diffusers','transformers','accelerate','numpy','scipy','safetensors','Pillow','clean-fid']
print(json.dumps({'python':sys.version,'packages':{n:importlib.metadata.version(n) for n in names}},indent=2))
PY
