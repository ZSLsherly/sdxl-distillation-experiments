from imaginaire.utils.config_helper import override
from imaginaire.lazy_config import LazyCall as L
from sparkdiffusion.configs.registry_distill import make_config as base_config
from smoke_metrics import SmokeMetrics
def make_config():
 root='/data5/shenglin.zeng'
 model=root+'/models/Wan2.1-T2V-1.3B'
 c=override(base_config(),['--',
  'experiment=wan2pt1_1pt3B_res480p_t2v_rola_distill',
  'net=wan2pt1_1pt3B_t2v','optimizer=adamw','optimizer_fake_score=adamw','callbacks=[basic]',
  '++model.config.net.use_rola_attn=False',
  'job.group=CrossDistill_Smoke','job.name=wan1.3B_dense_pcm_dmd_smoke',
  'model.config.fsdp_shard_size=4','model.config.loss_scale_dmd=1.0','model.config.loss_scale_tdm=0.0',
  'model.config.ema.enabled=False','model.config.backward_timesteps=[0.94,0.852895,0.608979]',
  'model.config.rf_t_max=0.94','model.config.p_D.shift=1.0',
  f'model.config.teacher_ckpt={model}',f'model.config.student_ckpt={model}',
  f'model.config.tokenizer.vae_pth={model}/Wan2.1_VAE.pth',
  f'model.config.text_encoder_path={model}/models_t5_umt5-xxl-enc-bf16.pth',
  f'model.config.tokenizer_path={model}/google/umt5-xxl',f'model.config.neg_embed_path={root}/models/negative_embed.pt',
  'trainer.max_iter=10','trainer.logging_iter=1','trainer.grad_accum_iter=1',
  'scheduler.warm_up_steps=[2]',
  'checkpoint.save_iter=1000',
  f'dataloader_train.tar_path_pattern={root}/datasets/smoke_1.3B_480p_81f_8step/shard*.tar',
  'dataloader_train.batch_size=1','dataloader_train.num_workers=1','dataloader_train.shuffle_buffer=4',
 ])
 # Keep numerical/precision callbacks; omit sampling, remote logging and periodic GC for timing.
 c.trainer.callbacks={k:v for k,v in c.trainer.callbacks.items() if k in ['low_prec','grad_clip']}
 c.trainer.callbacks['smoke_metrics']=L(SmokeMetrics)()
 c.defaults=['_self_']
 return c
