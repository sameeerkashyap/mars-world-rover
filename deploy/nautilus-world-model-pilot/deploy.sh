#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
kube=(rtk kubectl --context nautilus -n razvanlab)
name=mars-rover-world-model-pilot-20261121-v1
existing="$("${kube[@]}" get job "$name" --ignore-not-found -o name)"
if [[ -n "$existing" ]]; then
    echo "$name already exists; inspect it before launching another run." >&2
    exit 1
fi
"${kube[@]}" create configmap "$name-source" \
    --from-file="setup.sh=$root/deploy/nautilus-world-model-pilot/setup.sh" \
    --from-file="base_setup.sh=$root/deploy/nautilus-jepa-rover/setup.sh" \
    --from-file="train.sh=$root/deploy/nautilus-world-model-pilot/train.sh" \
    --from-file="prepare_split.py=$root/deploy/nautilus-jepa-rover/prepare_split.py" \
    --from-file="train_world_model_pilot.py=$root/scripts/train_world_model_pilot.py" \
    --from-file="evaluate_world_model_pilot.py=$root/scripts/evaluate_world_model_pilot.py" \
    --from-file="train_rover_jepa.py=$root/scripts/train_rover_jepa.py" \
    --from-file="train_rover_baseline.py=$root/scripts/train_rover_baseline.py" \
    --from-file="rover_training.py=$root/src/rover_training.py" \
    --from-file="rover_world_model.py=$root/src/rover_world_model.py" \
    --from-file="vit.py=$root/external/jepa-wms/app/plan_common/models/vit.py" \
    --from-file="AdaLN_vit.py=$root/external/jepa-wms/app/plan_common/models/AdaLN_vit.py" \
    --from-file="modules.py=$root/external/jepa-wms/src/models/utils/modules.py" \
    --from-file="logging.py=$root/external/jepa-wms/src/utils/logging.py" \
    --from-file="tensors.py=$root/external/jepa-wms/src/utils/tensors.py" \
    --from-file="upstream.json=$root/notes/upstream.json"
"${kube[@]}" patch configmap "$name-source" --type=merge -p '{"immutable":true}'
"${kube[@]}" apply --dry-run=server -f "$root/deploy/nautilus-world-model-pilot/job.yaml"
"${kube[@]}" create -f "$root/deploy/nautilus-world-model-pilot/job.yaml"
