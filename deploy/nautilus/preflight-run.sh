#!/usr/bin/env bash
set -euo pipefail
mode="${1:?Usage: preflight-run.sh rover|jepa|data}"
case "$mode" in rover|jepa|data) ;; *) exit 2 ;; esac
bash /opt/cluster/runtime-setup.sh "$mode"
report_dir="/data/runs/setup/${HOSTNAME}"
uv pip check 2>&1 | tee "$report_dir/dependency-check.txt"
case "$mode" in
    rover)
        cd "$ROVER_LAB_PATH"
        python tools/verify_stack.py --strict-pip-check > "$report_dir/stack.json"
        test -s rover_envs/assets/terrains/mars/terrain1/terrain_only.usd
        ;;
    jepa)
        cd /runtime/jepa-wms
        MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa python -c 'import torchcodec; import mujoco_py'
        python -c 'import torch; model, preprocessor = torch.hub.load("/runtime/jepa-wms", "jepa_wm_pointmaze", source="local", device="cpu"); print(type(model).__name__, type(preprocessor).__name__)'
        ;;
    data)
        cd /runtime/jepa-wms
        python src/scripts/download_data.py --list
        ;;
esac
touch "$report_dir/PREFLIGHT_SUCCESS"
echo 'Local installation preflight passed. GPU simulation has not been tested.'
