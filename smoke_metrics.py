import json,time,math
from pathlib import Path
import torch
from imaginaire.utils.callback import Callback
class SmokeMetrics(Callback):
 def on_training_step_start(self,model,data,iteration=0):
  torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();self.started=time.perf_counter()
  net=model.net if model.is_student_phase(iteration) else model.net_fake_score
  self.probe=next(net.parameters()).detach()
  if hasattr(self.probe,'to_local'):self.probe=self.probe.to_local()
  self.before=self.probe.reshape(-1)[:1024].clone()
 def on_training_step_end(self,model,data_batch,output_batch,loss,iteration=0):
  torch.cuda.synchronize()
  result={'iteration':iteration,'rank':torch.distributed.get_rank(),'phase':'student' if 'loss_dmd' in output_batch else 'critic','seconds':time.perf_counter()-self.started,'loss':float(loss.detach().sum()),'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'peak_reserved_GiB':torch.cuda.max_memory_reserved()/1024**3}
  result['parameter_delta_max']=float((self.probe.reshape(-1)[:1024]-self.before).abs().max())
  for key in ['loss_dmd','loss_div','loss_pcm_div','loss_fake']:
   if key in output_batch:
    value=output_batch[key];result[key]=float(value.detach().sum()) if torch.is_tensor(value) else value
  assert math.isfinite(result['loss']),result
  path=Path(self.config.job.path_local)/f'metrics_rank{result["rank"]}.jsonl'
  with path.open('a') as f:f.write(json.dumps(result)+'\n')
  print('SMOKE_METRIC '+json.dumps(result),flush=True)
