set -eu
BASE=/data1/shenglin.zeng/dymd_prior_8_20261101
export http_proxy=http://127.0.0.1:18897 https_proxy=http://127.0.0.1:18897
export HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=180
/data1/shenglin.zeng/miniconda3/bin/conda create -p "$BASE/env" python=3.11 pip -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
"$BASE/env/bin/pip" install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
"$BASE/env/bin/pip" install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.7-cp311-cp311-linux_x86_64.whl
"$BASE/env/bin/pip" install -r "$BASE/official/pf_wan/requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple/
"$BASE/env/bin/pip" freeze > "$BASE/environment.txt"
touch "$BASE/ENV_READY"
