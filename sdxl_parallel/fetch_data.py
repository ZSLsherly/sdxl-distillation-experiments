"""Fetch the next official LAION latent shard using verified source metadata."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests

root = Path('/data5/shenglin.zeng/sdxl_crossdistill')
meta = json.loads(Path(__file__).with_name('data_source.json').read_text())
target = root / 'datasets/dmd2' / Path(meta['file']).name
if target.exists():
    with target.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    if target.stat().st_size == meta['size'] and digest == meta['sha256']:
        print('DATA_SHARD_ALREADY_VERIFIED', flush=True)
        raise SystemExit(0)
parts = target.with_suffix('.download_parts')
parts.mkdir(exist_ok=True)
chunk = 32 * 1024**2
bounds = [(start, min(start + chunk, meta['size']) - 1) for start in range(0, meta['size'], chunk)]

def fetch(bounds):
    start, end = bounds
    path = parts / str(start)
    if path.exists() and path.stat().st_size == end - start + 1:
        return
    session = requests.Session()
    session.trust_env = False
    for attempt in range(3):
        try:
            with session.get(meta['location'], headers={'Range': f'bytes={start}-{end}'},
                             stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                if response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {start}-{end}/'):
                    raise RuntimeError('Unexpected HTTP range response')
                with path.with_suffix('.partial').open('wb') as f:
                    for block in response.iter_content(1024**2):
                        f.write(block)
            if path.with_suffix('.partial').stat().st_size != end - start + 1:
                raise RuntimeError('Incomplete range')
            path.with_suffix('.partial').replace(path)
            return
        except Exception as error:
            print('RANGE_RETRY', start, attempt, type(error).__name__, flush=True)
            if attempt == 2:
                raise
            time.sleep(2)

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    for completed, future in enumerate(concurrent.futures.as_completed([pool.submit(fetch, b) for b in bounds]), 1):
        future.result()
        print('FETCH_PROGRESS', completed, len(bounds), flush=True)
digest = hashlib.sha256()
temp = target.with_suffix('.partial')
with temp.open('wb') as f:
    for start, _ in bounds:
        with (parts / str(start)).open('rb') as part:
            while block := part.read(4 * 1024**2):
                f.write(block)
                digest.update(block)
if temp.stat().st_size != meta['size'] or digest.hexdigest() != meta['sha256']:
    raise RuntimeError('Shard checksum mismatch')
temp.replace(target)
print('DATA_SHARD_VERIFIED', target.name, meta['size'], digest.hexdigest(), flush=True)
