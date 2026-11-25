export http_proxy=http://127.0.0.1:18897 https_proxy=http://127.0.0.1:18897
/data1/shenglin.zeng/miniconda3/envs/sherly/bin/python - <<'PY'
from huggingface_hub import get_hf_file_metadata,hf_hub_url
for f in ['backbone.pth','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth']:
 m=get_hf_file_metadata(hf_hub_url('DAGroup-PKU/PF_Wan',f,revision='8fa7e17d5f944d0275df7dc7929137b956367ff8'))
 print(f,m.size,m.etag,flush=True)
PY
ps -u shenglin.zeng -o pid,ppid,etime,args | grep -E 'snapshot|install.sh|download.sh|pip install|conda create|python -' | head -20
du -sh /data1/shenglin.zeng/dymd_prior_8_20261101/official/pf_wan/checkpoints/PF_Wan
