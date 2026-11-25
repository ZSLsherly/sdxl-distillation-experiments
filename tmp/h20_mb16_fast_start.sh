set -eu
. /root/sdxl_research/env.sh
base=/root/autodl-tmp/h20_gan_mb32
d=/root/autodl-tmp/h20_gan_mb16
c="$d/code_fast";mkdir -p "$c";cp "$d/code"/*.py "$d/code/config.json" "$c/"
cd "$c"
python - <<'PY'
from pathlib import Path
p=Path('sam_pixel.py');s=p.read_text().replace('self.backbone = backbone.eval().requires_grad_(False)','self.backbone = backbone.eval().requires_grad_(False).to(torch.bfloat16)')
a=s.index('        def encode(z):');b=s.index('    def forward(',a)
s=s[:a]+'''        # Native full-image/full-batch attention. Keep only the four necessary feature taps.
        with torch.autocast("cuda",dtype=torch.bfloat16):
            x=self.backbone.patch_embed(x.to(torch.bfloat16))
            if self.backbone.pos_embed is not None:x=x+self.backbone.pos_embed
            features=[]
            for i,layer in enumerate(self.backbone.layers,1):
                x=checkpoint(layer,x,use_reentrant=False) if x.requires_grad else layer(x)
                if i in (3,6,9,12):features.append(x.permute(0,3,1,2).contiguous().float())
        return tuple(features)

'''+s[b:]
p.write_text(s)
p=Path('train.py');s=p.read_text()
key='    def predict(model, x, t, condition, negative=False, score_stage=None):'
s=s.replace(key,'''    def move_fake_moments(dst):
        torch.cuda.synchronize()
        begin=time.monotonic()
        for state in of.state.values():
            for k in ('exp_avg','exp_avg_sq'):
                if k in state: state[k]=state[k].to(dst)
        torch.cuda.synchronize()
        record(dict(phase='fake_adam_transfer',g=gcount+1,destination=str(dst),seconds=time.monotonic()-begin))

'''+key)
key='                of.step()';assert s.count(key)==1;s=s.replace(key,'                if fake_repeat==0 and gcount>0: move_fake_moments(device)\n'+key)
key='        og.zero_grad(set_to_none=True)\n        tick = time.monotonic()';assert s.count(key)==1;s=s.replace(key,'        move_fake_moments("cpu")\n'+key)
p.write_text(s)
p=Path('run_h20_16.py');p.write_text(p.read_text().replace('h20_gan_mb16_a4','h20_gan_mb16_fast'))
PY
L="$base/blas126/nvidia/cublas/lib";export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12" PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
python -m py_compile train.py sam_pixel.py
nohup python -u run_h20_16.py </dev/null > h20.log 2>&1 & echo $!
