set -eu
ROOT=/data5/shenglin.zeng
mkdir -p "$ROOT"/{envs,models,cache,tmp,logs,benchmarks,src,datasets,outputs}
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip" HF_HOME="$ROOT/cache/huggingface" XDG_CACHE_HOME="$ROOT/cache"
printf 'SYSTEM\n'
uname -a
df -h "$ROOT"
free -h
nvidia-smi
nvidia-smi topo -m
printf '\nTOOLS\n'
command -v git gcc g++ nvcc fio || true
printf '\nCUDA\n'
ls -d /usr/local/cuda* 2>/dev/null || true
printf '\nNETWORK\n'
for host in https://github.com https://huggingface.co https://pypi.org; do curl -IL --max-time 15 -s "$host" | head -3; done
printf '\nREPOSITORY\n'
if [ ! -d "$ROOT/src/SparkDiffusion/.git" ]; then git clone --depth 1 https://github.com/AlibabaResearch/SparkDiffusion.git "$ROOT/src/SparkDiffusion"; fi
cat "$ROOT/src/SparkDiffusion/AGENTS.md"
find "$ROOT/src/SparkDiffusion/.agents/skills" -name SKILL.md -print
