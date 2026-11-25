for u in https://hf-mirror.com/api/models/DAGroup-PKU/PF_Wan https://huggingface.co/api/models/DAGroup-PKU/PF_Wan; do
 env -u HTTPS_PROXY -u HTTP_PROXY -u http_proxy -u https_proxy curl -4 -sSL --connect-timeout 5 --max-time 10 -o /dev/null -w '%{http_code} %{speed_download}\n' "$u"
done
