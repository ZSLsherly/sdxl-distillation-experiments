"""Atomic JSON publication tolerant of brief Windows reader sharing locks."""
import json,os,time
from pathlib import Path

def read_json(path):
 for attempt in range(500):
  try:return json.loads(Path(path).read_text(encoding='utf-8'))
  except PermissionError:
   if attempt==499:raise
   time.sleep(.01)

def atomic_json(path,value):
 path=Path(path)
 temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
 temp.write_text(json.dumps(value),encoding='utf-8')
 for attempt in range(500):
  try:
   temp.replace(path)
   return
  except PermissionError:
   if attempt==499:raise
   time.sleep(.01)
