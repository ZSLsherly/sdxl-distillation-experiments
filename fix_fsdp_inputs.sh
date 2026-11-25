set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT/src/SparkDiffusion"
python - <<'PY'
from pathlib import Path
p=Path('sparkdiffusion/models/t2v_model_distill.py');s=p.read_text();s=s.replace('MixedPrecisionPolicy(param_dtype=self.precision, reduce_dtype=torch.float32)','MixedPrecisionPolicy(param_dtype=self.precision, reduce_dtype=torch.float32, cast_forward_inputs=False)');p.write_text(s)
PY
