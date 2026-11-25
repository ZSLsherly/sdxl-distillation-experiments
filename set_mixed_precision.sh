set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT/src/SparkDiffusion"
python - <<'PY'
from pathlib import Path
p=Path('sparkdiffusion/models/t2v_model_distill.py');s=p.read_text()
s=s.replace('MixedPrecisionPolicy(reduce_dtype=torch.float32)','MixedPrecisionPolicy(param_dtype=self.precision, reduce_dtype=torch.float32)')
s=s.replace('self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)','self.net = self.net.to(memory_format=memory_format, device="cuda", dtype=torch.float32)')
s=s.replace('self.net_fake_score = self.net_fake_score.to(memory_format=memory_format, **self.tensor_kwargs)','self.net_fake_score = self.net_fake_score.to(memory_format=memory_format, device="cuda", dtype=torch.float32)')
p.write_text(s)
PY
python -m compileall -q sparkdiffusion/models/t2v_model_distill.py
git diff --stat
tail -5 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
