#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
test -f /runtime/ready-rover
if [[ "${ACCEPT_EULA:-}" != Y ]]; then
    echo 'Set ACCEPT_EULA=Y only after reviewing the NVIDIA Isaac Sim EULA.' >&2
    exit 2
fi
run_dir="/data/runs/rover/${HOSTNAME}"
mkdir -p "$run_dir" /data/runs/rover/training
exec > >(tee "$run_dir/console.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"; if [ -r /sys/fs/cgroup/memory.events ]; then cat /sys/fs/cgroup/memory.events > "$run_dir/memory.events"; fi' EXIT
cp /opt/upstream.json "$run_dir/"
cd "$ROVER_LAB_PATH"
export PYTHONPATH="$ROVER_LAB_PATH${PYTHONPATH:+:$PYTHONPATH}"
export ISAAC_SIM_PATH="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("isaacsim").submodule_search_locations)))')"
# Upstream training writes relative to this checkout.
nvidia-smi
python tools/verify_stack.py > "$run_dir/stack.json"
cat "$run_dir/stack.json"
printf '%s\n' "$@" > "$run_dir/arguments.txt"
# Always append the headless setting, including when overriding the demo command.
echo '[INFO] Launching headless rover example'
timeout --signal=TERM --kill-after=30s 900s python -X faulthandler -u "$@" --viz none
touch "$run_dir/SUCCESS"
