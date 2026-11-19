#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
test -f /runtime/ready-rover
run_dir="/data/runs/rover/${HOSTNAME}"
mkdir -p "$run_dir" /data/runs/rover/training
exec > >(tee "$run_dir/console.log") 2>&1
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/exit-code"' EXIT
cp /opt/collect/* "$run_dir/"
cd "$ROVER_LAB_PATH"
export PYTHONPATH="$ROVER_LAB_PATH${PYTHONPATH:+:$PYTHONPATH}"
export ISAAC_SIM_PATH="$(python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("isaacsim").submodule_search_locations)))')"
nvidia-smi > "$run_dir/hardware.txt"
python -c 'import json,torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()[0] >= 8; assert torch.zeros(1,device="cuda").item() == 0; print(json.dumps({"torch":str(torch.__version__),"gpu":torch.cuda.get_device_name()}))' > "$run_dir/cuda-preflight.json"
python tools/verify_stack.py > "$run_dir/stack.json"
for seed in 16 17 18 19 20 21; do
    terrain=mars
    if (( seed % 2 )); then terrain=debug; fi
    output="/data/datasets/rover-jepa-collection/seed-$seed-$terrain"
    python -u /opt/collect/rover_record_dataset.py \
        --seed "$seed" --terrain "$terrain" --varied --steps 3000 --output "$output" --viz none
    test -s "$output/$HOSTNAME/metadata.json"
    printf '%s %s\n' "$seed" "$terrain" >> "$run_dir/completed-recordings.txt"
done
touch "$run_dir/SUCCESS"
