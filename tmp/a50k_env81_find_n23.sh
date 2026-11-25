find /data1/shenglin.zeng /data3/shenglin.zeng -maxdepth 8 -name model_index.json -print 2>/dev/null | head -15
R=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
"$R/env/bin/python" - <<'PY'
import torch,diffusers
p='/data3/shenglin.zeng/sdxl_turning_ablation_20261028/datasets/parallel_10k_seed20261018/prepared.pt'
x=torch.load(p,map_location='cpu',mmap=True,weights_only=True)
print('VERSIONS',torch.__version__,diffusers.__version__)
print('DATA',[(k,tuple(v.shape) if torch.is_tensor(v) else len(v) if isinstance(v,list) else v) for k,v in x.items()])
PY
