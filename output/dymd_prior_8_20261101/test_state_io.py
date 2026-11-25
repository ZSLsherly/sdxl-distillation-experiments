import json,os,tempfile,threading,time
from pathlib import Path
from state_io import atomic_json,read_json
B=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='atomic_state_test_',dir=B) as folder:
 root=Path(folder).resolve();assert root.is_relative_to(B.resolve())
 p=root/'state.json';atomic_json(p,{'value':0})
 held=p.open();errors=[]
 def writer():
  try:atomic_json(p,{'value':1})
  except Exception as e:errors.append(e)
 t=threading.Thread(target=writer);t.start();time.sleep(.1)
 if os.name=='nt':assert t.is_alive(),'Expected the held Windows reader to block replacement'
 held.close();t.join(3)
 assert not t.is_alive() and not errors,errors
 assert json.loads(p.read_text())=={'value':1}
 stop=threading.Event()
 def reader():
  while not stop.is_set():
   try:
    read_json(p)
   except Exception as e:errors.append(e)
   time.sleep(.001)
 t=threading.Thread(target=reader);t.start()
 try:
  for i in range(2,102):atomic_json(p,{'value':i})
 finally:stop.set();t.join()
 assert not errors,errors
 assert json.loads(p.read_text())['value']==101
print('PASS: Windows open-reader conflict recovers; 100 atomic updates with concurrent readers remain valid')
