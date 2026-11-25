/data1/shenglin.zeng/miniconda3/envs/sherly/bin/python - <<'PY'
import sys, importlib.metadata as m
print(sys.version)
for x in ['torch','torchvision','flash_attn','xfuser','diffusers','transformers','huggingface_hub','numpy']:
 try: print(x,m.version(x))
 except: print(x,'MISSING')
PY
find /data1/shenglin.zeng/sdxl_eval_uniform_20261025/runtime -maxdepth 3 -name python -o -name python3
find /data1/shenglin.zeng -maxdepth 5 -type f \( -name '*VAE*' -o -name '*umt5*' -o -name '*backbone.pth' \) 2>/dev/null | head -30
