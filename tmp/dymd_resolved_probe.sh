for spec in 'hf-mirror.com:443:160.16.86.14' 'download.pytorch.org:443:52.85.128.85'; do
 domain=${spec%%:*}
 env -u HTTPS_PROXY -u HTTP_PROXY -u http_proxy -u https_proxy curl --resolve "$spec" -4 -sSL --connect-timeout 5 --max-time 12 -o /dev/null -w "$domain %{http_code} %{speed_download}\n" "https://$domain/"
done
