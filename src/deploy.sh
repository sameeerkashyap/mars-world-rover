#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
kube=(rtk kubectl --context nautilus -n razvanlab)
# Refuse to change source mounted by an existing run.
existing="$("${kube[@]}" get job mars-jepa-step1 --ignore-not-found -o name)"
if [[ -n "$existing" ]]; then
    echo 'mars-jepa-step1 already exists. Inspect its results before removing it.' >&2
    exit 1
fi
# Upsert the HuggingFace token secret from the local .env (gitignored) so the
# gated pointmaze dataset download can authenticate. Requires HF_TOKEN in .env.
"${kube[@]}" create secret generic hf-token \
    --from-env-file="$root/.env" \
    --dry-run=client -o yaml | "${kube[@]}" apply -f -
"${kube[@]}" create configmap mars-jepa-step1-source \
    --from-file="$root/src/step1/run.sh" \
    --from-file="$root/src/step1/jepa_smoke.py" \
    --from-file="$root/src/step1/jepa_plan.py" \
    --from-file="$root/src/step1/inspect_data.py" \
    --from-file="$root/deploy/nautilus/runtime-setup.sh" \
    --from-file="$root/notes/upstream.json" \
    --dry-run=client -o yaml | "${kube[@]}" apply -f -
"${kube[@]}" apply --dry-run=server -f "$root/src/step1/job.yaml"
"${kube[@]}" apply -f "$root/src/step1/job.yaml"
