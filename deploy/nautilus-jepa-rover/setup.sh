#!/usr/bin/env bash
set -euo pipefail
run_dir="/data/runs/setup/$HOSTNAME"
mkdir -p "$run_dir" /runtime
exec > >(tee "$run_dir/install.log") 2>&1
uv venv --python 3.12 --seed "$VIRTUAL_ENV"
uv pip install torch==2.10.0 torchvision==0.25.0 numpy==2.3.1 h5py==3.15.1 einops==0.8.1
git init /runtime/dinov2
git -C /runtime/dinov2 remote add origin https://github.com/facebookresearch/dinov2.git
git -C /runtime/dinov2 fetch --depth 1 origin 7764ea0f912e53c92e82eb78a2a1631e92725fc8
git -C /runtime/dinov2 checkout --detach FETCH_HEAD
test "$(git -C /runtime/dinov2 rev-parse HEAD)" = 7764ea0f912e53c92e82eb78a2a1631e92725fc8
python -c 'import torch; torch.hub.load("/runtime/dinov2", "dinov2_vits14", source="local", pretrained=True)'
uv pip freeze > "$run_dir/packages.txt"
cp /opt/train/upstream.json "$run_dir/"
uv cache clean
touch /runtime/ready-rover-jepa
