set -eu
url=https://pjlab-shanghai-moce.oss-cn-shanghai.aliyuncs.com/static/installpack/discovery-ctl/prod/v1.2.1/install.sh
if [ ! -x "$HOME/.discovery/bin/discovery-ctl" ]; then
  f=$(mktemp)
  trap 'rm -f "$f"' EXIT
  curl -fsSL --max-time 90 "$url" -o "$f"
  sh "$f"
fi
"$HOME/.discovery/bin/discovery-ctl" login --help
"$HOME/.discovery/bin/discovery-ctl" dataset --help
