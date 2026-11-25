set -eu
cd /data5/shenglin.zeng/checkpoints/h20_gan_20261104
test "$(sha256sum student_rank0.pt.partial | cut -d' ' -f1)" = 361ad217704fe6b44d01242ea7b91e6ce4c63c90090e40197da99edd0266e068
mv student_rank0.pt.partial student_rank0.pt
echo VERIFIED_LAB_WEIGHT
