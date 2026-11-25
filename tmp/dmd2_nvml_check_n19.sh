R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
"$R/env/bin/python" - <<'PY'
import importlib.util
print('PYNVML_AVAILABLE',importlib.util.find_spec('pynvml') is not None)
PY
