ROOT=/data5/shenglin.zeng
for endpoint in https://huggingface.co https://hf-mirror.com; do
 printf '\n%s\n' "$endpoint"
 curl -L -s --connect-timeout 5 --max-time 12 -r 0-1048575 -o "$ROOT/tmp/network-probe.bin" -w 'http=%{http_code} bytes=%{size_download} speed=%{speed_download} time=%{time_total}\n' "$endpoint/Wan-AI/Wan2.1-T2V-1.3B/resolve/main/diffusion_pytorch_model.safetensors?download=true"
done
printf '\nINSTALL STATUS\n'
tail -4 "$ROOT/logs/install_torch.log"
tail -4 "$ROOT/logs/install_deps.log"
