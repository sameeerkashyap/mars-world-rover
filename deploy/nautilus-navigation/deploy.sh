#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
checkpoint="${1:?Usage: deploy.sh /path/to/goal-motion-v1/model.pt}"
test -s "$checkpoint"
kube=(rtk kubectl --context nautilus -n razvanlab)
for job in mars-rover-navigation-v4 mars-rover-navigation-mars-v4; do
    existing="$("${kube[@]}" get job "$job" --ignore-not-found -o name)"
    if [[ -n "$existing" ]]; then
        echo "$job already exists. Inspect its saved results before removing it." >&2
        exit 1
    fi
done
"${kube[@]}" create configmap mars-rover-navigation-v4-source \
    --from-file="rover_navigate.py=$root/deploy/nautilus-navigation/rover_navigate.py" \
    --from-file="rover_plan.py=$root/scripts/rover_plan.py" \
    --from-file="train_rover_baseline.py=$root/scripts/train_rover_baseline.py" \
    --from-file="train_rover_jepa.py=$root/scripts/train_rover_jepa.py" \
    --from-file="rover_forecast.py=$root/src/rover_forecast.py" \
    --from-file="terrain_cost.py=$root/src/terrain_cost.py" \
    --from-file="evaluation_terrain.py=$root/src/evaluation_terrain.py" \
    --from-file="model.pt=$checkpoint" \
    --from-file="runtime-setup.sh=$root/deploy/nautilus/runtime-setup.sh" \
    --from-file="run.sh=$root/deploy/nautilus-navigation/run.sh" \
    --from-file="upstream.json=$root/notes/upstream.json" \
    --dry-run=client -o yaml | "${kube[@]}" apply -f -
"${kube[@]}" patch configmap mars-rover-navigation-v4-source --type=merge -p '{"immutable":true}'
for manifest in job.yaml mars-job.yaml; do
    "${kube[@]}" apply --dry-run=server -f "$root/deploy/nautilus-navigation/$manifest"
done
for manifest in job.yaml mars-job.yaml; do
    "${kube[@]}" apply -f "$root/deploy/nautilus-navigation/$manifest"
done
