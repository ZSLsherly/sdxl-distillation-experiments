set -eu
ls -la /home/shenglin.zeng
ls -ld /data*/shenglin.zeng 2>/dev/null || true
find /home/shenglin.zeng -maxdepth 5 -type f -name python 2>/dev/null | head -20
find /home/shenglin.zeng -maxdepth 4 -type d \( -name '*wan*' -o -name '*codex*' -o -name envs \) 2>/dev/null | head -30
python3 -c 'import sys; print(sys.version)' || true
curl -IL --connect-timeout 10 --max-time 20 https://huggingface.co/DAGroup-PKU/PF_Wan/resolve/main/config.json 2>&1 | tail -8
curl -IL --connect-timeout 10 --max-time 20 https://github.com/DAGroup-PKU/PhysisForcing 2>&1 | tail -8
