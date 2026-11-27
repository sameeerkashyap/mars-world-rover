#!/usr/bin/env bash
set -euo pipefail
bash /opt/train/base_setup.sh
uv pip install timm==1.0.19
uv pip freeze > "/data/runs/setup/$HOSTNAME/packages.txt"
touch /runtime/ready-rover-pilot
