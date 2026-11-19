#!/usr/bin/env bash
set -euo pipefail
test -f /runtime/ready-jepa
run_dir="$JEPAWM_LOGS/$HOSTNAME"
mkdir -p "$run_dir"
exec > >(tee "$run_dir/console.log") 2>&1
cp /opt/upstream.json "$run_dir/"
cp /opt/step1/{run.sh,jepa_smoke.py,jepa_plan.py,inspect_data.py} "$run_dir/"
cd /runtime/jepa-wms
python setup_macros.py
nvidia-smi | tee "$run_dir/hardware.txt"
python /opt/step1/jepa_smoke.py
# The loaders (inspect_data.py and the planner's "PointMaze" path) expect the
# payload directly under $JEPAWM_DSET/point_maze, but upstream download_data.py
# (unzip_in_place) nests it under point_maze/point_maze/ whenever the target dir
# already exists. Flatten any nesting, (re)download only if the payload is truly
# missing, then flatten again in case the download re-nested.
pm="$JEPAWM_DSET/point_maze"
flatten_pm() {
    if [ -f "$pm/point_maze/states.pth" ] && [ ! -f "$pm/states.pth" ]; then
        mv "$pm/point_maze/"* "$pm/" && rmdir "$pm/point_maze"
    fi
}
flatten_pm
dl_force=""
[ -f "$pm/states.pth" ] || dl_force="--force"
python src/scripts/download_data.py --dataset pointmaze $dl_force
flatten_pm
test -f "$pm/states.pth"  # fail loudly if the payload still is not in place
python /opt/step1/inspect_data.py
python /opt/step1/jepa_plan.py --episodes 2
touch "$run_dir/SUCCESS"
