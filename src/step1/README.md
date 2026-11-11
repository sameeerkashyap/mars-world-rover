# Step 1: reference world-model planning

Run the checkpoint check, download PointMaze data, read one complete sample,
and run two planning episodes with seed 0. This checks the pipeline. It does
not train a rover model or reproduce the full paper benchmark.

Current status: the first Nautilus run failed during dependency setup because
the upstream Metaworld `master` branch is missing. See
[results and the setup blocker](../../docs/results.md#reference-planning)
before retrying. No reference planning metrics are available yet.

From the repository root:

```bash
rtk proxy bash src/step1/deploy.sh
rtk kubectl --context nautilus -n razvanlab get pods -l job-name=mars-jepa-step1
rtk kubectl --context nautilus -n razvanlab logs -f job/mars-jepa-step1 -c setup
rtk kubectl --context nautilus -n razvanlab logs -f job/mars-jepa-step1 -c jepa
```

The job uses the existing `mars-world-data` storage and published `uv7` image.
The deploy script mounts the current source, shared runtime installer, and
upstream revision file through a ConfigMap. No image rebuild is needed.
It refuses to replace an existing job. Review a finished job before deleting
it and running the script again. Never delete the shared storage to retry.

The job requests one CUDA GPU, four CPUs, 16 GiB RAM, and 40 GiB temporary disk.
It stops after four hours and has no automatic retries. Setup installs JEPA,
FFmpeg 7, and legacy MuJoCo 2.1 in a temporary Python 3.10 environment.

Outputs persist in `/data/runs/jepa/<pod-name>/`:

- `console.log`, `hardware.txt`, `upstream.json`, and copies of run source.
- `model.json` and `model-config.yaml`: checkpoint/preprocessor loading evidence.
- `dataset.json`: inspected sample shapes and source.
- `pointmaze.yaml` and upstream evaluation logs/metrics.
- `SUCCESS`: written only after all checks return exit code zero.

Dependency versions and install logs are under `/data/runs/setup/<pod-name>/`.
Read the planning metrics before accepting Step 1. A low success rate can still
show that the reduced evaluation works; it does not show navigation quality.
The planner uses three CEM iterations and 32 samples per iteration, with optional
image/state decoder heads disabled. CEM selects actions by sampling candidate
action sequences and keeping those with the lowest predicted goal error.
