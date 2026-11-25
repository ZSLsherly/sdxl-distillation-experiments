import sys, pathlib
sys.path.insert(0, '.image_runtime_pkgs')
import requests
p=pathlib.Path('benchmark_vendor');p.mkdir(exist_ok=True)
r=requests.get('https://api.github.com/repos/XPixelGroup/BasicSR/commits/master',timeout=30);r.raise_for_status();sha=r.json()['sha']
(p/'source_commit.txt').write_text(sha)
for src,name in [('basicsr/metrics/niqe.py','niqe.py'),('basicsr/metrics/metric_util.py','metric_util.py'),('basicsr/utils/matlab_functions.py','matlab_functions.py'),('basicsr/utils/color_util.py','color_util.py'),('basicsr/metrics/niqe_pris_params.npz','niqe_pris_params.npz'),('LICENSE.txt','LICENSE.txt')]:
    r=requests.get(f'https://raw.githubusercontent.com/XPixelGroup/BasicSR/{sha}/{src}',timeout=40);r.raise_for_status();(p/name).write_bytes(r.content)
    print(name,len(r.content))
print('COMMIT',sha)
r=requests.get(f'https://raw.githubusercontent.com/XPixelGroup/BasicSR/{sha}/test_scripts/data/baboon.png',timeout=40);r.raise_for_status();(p/'baboon.png').write_bytes(r.content)
s=(p/'niqe.py').read_text()
s=s[:s.index('@METRIC_REGISTRY.register()')]
s=s.replace('import cv2\n','').replace('from basicsr.metrics.metric_util import reorder_image, to_y_channel\n','').replace('from basicsr.utils.matlab_functions import imresize','from matlab_functions import imresize').replace('from basicsr.utils.registry import METRIC_REGISTRY\n','')
(p/'niqe_core.py').write_text(s)
import tarfile
with tarfile.open('benchmark_vendor.tar.gz','w:gz') as tf:
    tf.add(p,arcname='benchmark_vendor')
