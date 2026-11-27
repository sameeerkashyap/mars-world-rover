#!/usr/bin/env bash
set -euo pipefail
bash /opt/navigation/runtime-setup.sh rover
uv pip install h5py==3.15.1
uv pip freeze > "/data/runs/setup/$HOSTNAME/packages.txt"
