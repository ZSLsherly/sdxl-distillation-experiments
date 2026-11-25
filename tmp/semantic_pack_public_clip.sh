set -eu
cd /root/autodl-tmp/sdxl_runs/semantic_abc_20261103/model
test "$(cat revision.txt)" = 3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
test "$(sha256sum pytorch_model.bin | cut -d' ' -f1)" = a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f
tar -cf /root/autodl-tmp/public_clip_B32_semantic_20261104.tar -- config.json preprocessor_config.json tokenizer_config.json vocab.json merges.txt pytorch_model.bin revision.txt
