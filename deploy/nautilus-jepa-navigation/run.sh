#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
test -f /runtime/ready-rover-forecast
run_dir="/data/runs/rover-jepa-navigation/$HOSTNAME"
mkdir -p "$run_dir" /data/runs/rover/training
exec > >(tee "$run_dir/console.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT
cp -rL /opt/navigation "$run_dir/source"
cd "$ROVER_LAB_PATH"
export PYTHONPATH="/opt/navigation:$ROVER_LAB_PATH${PYTHONPATH:+:$PYTHONPATH}"
export ISAAC_SIM_PATH="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("isaacsim").submodule_search_locations)))')"
nvidia-smi > "$run_dir/hardware.txt"
python tools/verify_stack.py > "$run_dir/stack.json"
common=(--jepa-run /data/runs/rover-jepa/mars-rover-jepa-train-v12-dfj2t
        --terrain-checkpoint /data/runs/rover-jepa-scale/mars-rover-scale-20261120-v1/terrain-cost-mars-rover-scale-20261120-v1-s9ckw/best.pt
        --test-plan /opt/navigation/test-plan.json --viz none)
evaluate() {
    local phase="$1" terrain="$2"
    shift 2
    local output="$run_dir/$phase-$terrain"
    python -X faulthandler -u /opt/navigation/rover_navigate.py \
        "${common[@]}" --phase "$phase" --terrain "$terrain" --seeds "$@" --output "$output"
    test -f "$output/$HOSTNAME/SUCCESS"
}
previous=/data/runs/rover-jepa-navigation/mars-rover-jepa-navigation-20261120-v1-7hntv
for terrain in debug mars; do
    test -f "$previous/development-$terrain/mars-rover-jepa-navigation-20261120-v1-7hntv/SUCCESS"
    ln -s "$previous/development-$terrain" "$run_dir/development-$terrain"
done
evaluate fresh test-rocks-501 30 31 32
evaluate fresh test-rocks-502 33 34 35
python /opt/navigation/verify.py "$run_dir"
touch "$run_dir/SUCCESS"
