#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
kube=(rtk kubectl --context nautilus -n razvanlab)
name=mars-rover-jepa-navigation-20261120-v2
existing="$("${kube[@]}" get job "$name" --ignore-not-found -o name)"
if [[ -n "$existing" ]]; then
    echo "$name already exists; inspect its results before another launch." >&2
    exit 1
fi
"${kube[@]}" create configmap "$name-source" \
    --from-file="rover_navigate.py=$root/deploy/nautilus-navigation/rover_navigate.py" \
    --from-file="rover_plan.py=$root/scripts/rover_plan.py" \
    --from-file="train_rover_baseline.py=$root/scripts/train_rover_baseline.py" \
    --from-file="train_rover_jepa.py=$root/scripts/train_rover_jepa.py" \
    --from-file="rover_forecast.py=$root/src/rover_forecast.py" \
    --from-file="terrain_cost.py=$root/src/terrain_cost.py" \
    --from-file="evaluation_terrain.py=$root/src/evaluation_terrain.py" \
    --from-file="meta_vit.py=$root/external/jepa-wms/app/plan_common/models/vit.py" \
    --from-file="model.pt=$root/runs/rover-baseline/20261113T210435968626Z/model.pt" \
    --from-file="runtime-setup.sh=$root/deploy/nautilus/runtime-setup.sh" \
    --from-file="setup.sh=$root/deploy/nautilus-jepa-navigation/setup.sh" \
    --from-file="run.sh=$root/deploy/nautilus-jepa-navigation/run.sh" \
    --from-file="verify.py=$root/deploy/nautilus-jepa-navigation/verify.py" \
    --from-file="test-plan.json=$root/deploy/nautilus-jepa-navigation/test-plan.json" \
    --from-file="upstream.json=$root/notes/upstream.json"
"${kube[@]}" patch configmap "$name-source" --type=merge -p '{"immutable":true}'
"${kube[@]}" apply --dry-run=server -f "$root/deploy/nautilus-jepa-navigation/job.yaml"
"${kube[@]}" create -f "$root/deploy/nautilus-jepa-navigation/job.yaml"
