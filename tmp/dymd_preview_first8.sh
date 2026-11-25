set -eu
/data1/shenglin.zeng/dymd_prior_8_20261101/env/bin/python - <<'PY'
import pathlib,time,subprocess,json
b=pathlib.Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
deadline=time.monotonic()+300
while not all((b/'profiles'/f'trajectory_{i:02d}.complete').exists() for i in range(8)):
 if time.monotonic()>deadline:raise TimeoutError('First eight not complete within preview wait; sampling left running')
 time.sleep(10)
with (b/'analysis_preview_008.log').open('w') as log:
 subprocess.run([str(b/'env/bin/python'),str(b/'analyze.py'),str(b),'8'],stdout=log,stderr=subprocess.STDOUT,check=True)
s=json.loads((b/'analysis_preview_008/summary.json').read_text())
print(json.dumps({'n':s['n'],'preliminary':s['preliminary'],'metrics':{k:{'peak_sigma':v['mean_curve_peak_sigma'],'sample_peaks':v['per_sample_peak_sigma'],'prior_mass_sigma_ge_09_measured':v['unsmoothed_squared_mean_prior_mass_sigma_ge_0_9_measured_support'],'high_to_rest_mean_rate':v['high_region_mean_rate']/v['rest_region_mean_rate']} for k,v in s['metrics'].items()}},indent=2))
PY
