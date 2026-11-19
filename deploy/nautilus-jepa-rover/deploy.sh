#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
kube=(kubectl --context nautilus -n razvanlab)
for name in mars-rover-jepa-train-v12 mars-rover-jepa-collect-v7; do
    if [[ -n "$("${kube[@]}" get job "$name" --ignore-not-found -o name)" ]]; then
        echo "$name exists. Inspect its results before another launch." >&2
        exit 2
    fi
done
"${kube[@]}" create configmap mars-rover-jepa-train-v12-source \
    --from-file="setup.sh=$root/deploy/nautilus-jepa-rover/setup.sh" \
    --from-file="train.sh=$root/deploy/nautilus-jepa-rover/train.sh" \
    --from-file="prepare_split.py=$root/deploy/nautilus-jepa-rover/prepare_split.py" \
    --from-file="meta_vit.py=$root/external/jepa-wms/app/plan_common/models/vit.py" \
    --from-file="train_rover_jepa.py=$root/scripts/train_rover_jepa.py" \
    --from-file="train_rover_baseline.py=$root/scripts/train_rover_baseline.py" \
    --from-file="upstream.json=$root/notes/upstream.json"
"${kube[@]}" patch configmap mars-rover-jepa-train-v12-source --type=merge -p '{"immutable":true}'
"${kube[@]}" create configmap mars-rover-jepa-collect-v2-source \
    --from-file="collect.sh=$root/deploy/nautilus-jepa-rover/collect.sh" \
    --from-file="rover_record_dataset.py=$root/deploy/nautilus-training/rover_record_dataset.py" \
    --from-file="runtime-setup.sh=$root/deploy/nautilus/runtime-setup.sh" \
    --from-file="upstream.json=$root/notes/upstream.json"
"${kube[@]}" patch configmap mars-rover-jepa-collect-v2-source --type=merge -p '{"immutable":true}'
"${kube[@]}" apply --dry-run=server -f "$root/deploy/nautilus-jepa-rover/train-job.yaml"
"${kube[@]}" apply --dry-run=server -f "$root/deploy/nautilus-jepa-rover/collect-job.yaml"
"${kube[@]}" apply -f "$root/deploy/nautilus-jepa-rover/train-job.yaml"
"${kube[@]}" apply -f "$root/deploy/nautilus-jepa-rover/collect-job.yaml"
