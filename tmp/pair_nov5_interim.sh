. /root/sdxl_research/env.sh
cd "$SDXL_WORK/semantic_pair_1024_20261105"
grep "^A_pcm10 {" worker.log | cut -c1-250
