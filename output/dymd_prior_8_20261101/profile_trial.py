"""Use the existing sampler/profiler; add save timing and per-rank peak memory."""
import json,os,runpy,time
from pathlib import Path
import torch
from project.utils import common
B=Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
G=Path(os.environ['DYMD_TRIAL_DIR'])
save=common.save_video
def timed_save(*args,**kwargs):
 result=save(*args,**kwargs)
 (G/'video_saved.json').write_text(json.dumps({'path':str(result),'saved_unix':time.time()}))
 return result
common.save_video=timed_save
try:
 runpy.run_path(str(B/'profile_official.py'),run_name='__main__')
finally:
 if torch.cuda.is_initialized():
  (G/f'rank{os.environ.get("LOCAL_RANK","0")}_memory.json').write_text(json.dumps({'allocated_peak_bytes':torch.cuda.max_memory_allocated(),'reserved_peak_bytes':torch.cuda.max_memory_reserved(),'world_size':int(os.environ['WORLD_SIZE'])}))
