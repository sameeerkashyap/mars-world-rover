#!/usr/bin/env bash
set -euo pipefail
bash /opt/navigation/runtime-setup.sh rover
uv pip install einops==0.8.1 h5py==3.15.1
git init /runtime/dinov2
git -C /runtime/dinov2 remote add origin https://github.com/facebookresearch/dinov2.git
git -C /runtime/dinov2 fetch --depth 1 origin 7764ea0f912e53c92e82eb78a2a1631e92725fc8
git -C /runtime/dinov2 checkout --detach FETCH_HEAD
test "$(git -C /runtime/dinov2 rev-parse HEAD)" = 7764ea0f912e53c92e82eb78a2a1631e92725fc8
python -c 'import torch; torch.hub.load("/runtime/dinov2", "dinov2_vits14", source="local", pretrained=True)'
uv pip freeze > "/data/runs/setup/$HOSTNAME/forecast-packages.txt"
touch /runtime/ready-rover-forecast
