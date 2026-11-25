"""Entry point for the explicitly approved shared-GPU short test only."""
import os,runpy
from pathlib import Path
import torch
gpu=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(gpu)
torch.cuda.set_per_process_memory_fraction(14*2**30/torch.cuda.get_device_properties(gpu).total_memory,gpu)
runpy.run_path(str(Path(__file__).with_name('train.py')),run_name='__main__')
