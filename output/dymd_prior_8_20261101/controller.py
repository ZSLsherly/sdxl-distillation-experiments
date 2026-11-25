"""Single-experiment bounded controller: verify, wait for idle GPUs, run 128, analyze."""
import hashlib,json,os,subprocess,time,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
deadline=time.monotonic()+48*3600
def state(stage,**kw):
 d={'stage':stage,'target_trajectories':128,'updated_unix':time.time(),**kw}
 (B/'status.json').write_text(json.dumps(d,indent=2))
 print(json.dumps(d),flush=True)
def wait(predicate,stage):
 state(stage)
 while not predicate():
  if time.monotonic()>deadline:raise TimeoutError(stage)
  time.sleep(20)
def call(cmd,log):
 with (B/log).open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
def idle():
 global selected
 r=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu,memory.total','--format=csv,noheader,nounits'],text=True)
 rows={int(x.split(',')[0]):[int(y.strip()) for y in x.split(',')[1:]] for x in r.splitlines()}
 selected=[i for i,v in sorted(rows.items()) if v[0]<500 and v[1]==0 and v[2]>45000][:4]
 return len(selected)==4
try:
 wheels=['torch-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl','torchvision-0.22.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl','torchaudio-2.7.1+cu128-cp311-cp311-manylinux_2_28_x86_64.whl','flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl']
 wait(lambda:all((B/'wheels'/x).exists() for x in wheels),'waiting_for_verified_wheels')
 if not (B/'ENV_READY').exists():
  state('installing_environment');call(['bash',str(B/'install_from_wheels.sh')],'install_fast.log')
 files={'Wan2.1_VAE.pth':'38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981','backbone.pth':'569e41d3046fe8bdd8df8345a9715e2d8fc5d92d52811ac165ad8947d776a47b','models_t5_umt5-xxl-enc-bf16.pth':'7cace0da2b446bbbbc57d031ab6cf163a3d59b366da94e5afe36745b746fd81d'}
 weights=B/'official/pf_wan/checkpoints/PF_Wan'
 wait(lambda:all((weights/x).exists() for x in files),'waiting_for_model_weights')
 state('verifying_model_sha256')
 for name,expected in files.items():
  h=hashlib.sha256()
  with (weights/name).open('rb') as f:
   while data:=f.read(16*1024**2):h.update(data)
  assert h.hexdigest()==expected,(name,h.hexdigest())
 for name in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','spiece.model']:
  assert (weights/'google/umt5-xxl'/name).exists(),name
 (B/'MODEL_READY').touch()
 wait(idle,'waiting_for_four_idle_48gb_gpus')
 os.environ['DYMD_GPU_IDS']=','.join(map(str,selected))
 (B/'selected_gpus.json').write_text(json.dumps(selected))
 state('running_128_trajectories',gpus=selected);call(['bash',str(B/'run128.sh')],'run.log')
 state('analyzing');call([str(B/'env/bin/python'),str(B/'analyze.py'),str(B)],'analysis.log')
 state('complete',trajectories=128)
except Exception as e:
 state('failed',error=str(e));traceback.print_exc();raise
