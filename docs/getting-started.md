# Getting started

Use Nautilus for 3D rover simulation and reference JEPA evaluation. Use the
optional CPU prototype for local 2D data experiments. See the
[starter requirements](../REQUIREMENTS.md) for acceptance checks and the
[system overview](architecture.md) for the planned model pipeline.

Run all commands below from the repository root. They use the `rtk` command
proxy. Install RTK first, or remove its prefix when using these tools directly.

## Run on NRP Nautilus

For Step 1 (JEPA checkpoint loading, PointMaze sample inspection, and two planning
episodes), run `rtk proxy bash deploy/nautilus-reference/deploy.sh`. See the
[Step 1 guide](../deploy/nautilus-reference/README.md) for logs and saved results. The first run
failed on an upstream Metaworld branch reference. Resolve the
[setup blocker](results.md#reference-planning) before retrying.

See [the Nautilus deployment guide](../deploy/nautilus/README.md) for one slim shared
Docker image and separate Kubernetes Jobs for headless RLRoverLab, JEPA checkpoint
loading, dataset preparation, and PointMaze world-model planning. Results are
saved to a shared PVC. Build/publish the image and configure the manifests
before submitting. The A40 rover example is verified: 120 headless environment
steps, exit code 0. Use the deployment guide's `deploy/nautilus-a40` recipe.

## Local workspace

For rover JEPA training and simulation data collection, use the separate
[training guide](../deploy/nautilus-jepa-rover/README.md). It names the starting
model, fixed data splits, quality checks, and saved artifacts. Check existing
Jobs before launching; do not overwrite a prior run.

**Local hardware choice: CPU-only MarsExplorer on the Mac.** NVIDIA hardware
is not required for this local starter path. Use Nautilus RTX GPU nodes for
headless RLRoverLab/Isaac instead.

The lightweight environment has its own Python 3.10 environment, `.venv-rover`,
and a source snapshot at `external/MarsExplorer-Gymnasium`. Install and run with:

```bash
rtk proxy uv pip install --python .venv-rover/bin/python -e external/MarsExplorer-Gymnasium
rtk proxy .venv-rover/bin/python scripts/record_explorer.py
```

The last recorded local installation failed because uv could not resolve PyPI.
The recording script has been syntax checked; local execution remains unverified.
Once dependencies are installed, it saves `trajectory.npz`, `metadata.json`, and an SVG
path plot in `runs/explorer/` without opening a graphical window.

MarsExplorer is a 2D, four-action exploration environment. It supports early
data collection and predictive-model experiments; it does not simulate wheel
slip, traction, slopes, or 3D rover dynamics. Its task is coverage rather than
point-goal navigation, so it is a prototype milestone with its own limitations.

The workspace contains commit-pinned JEPA-WMs and RLRoverLab source snapshots
under `external/`, an isolated Python 3.10 environment at `.venv-jepa/`, and
directories for data, checkpoints, and runs. Snapshots have no Git history;
revision metadata is in [notes/upstream.json](../notes/upstream.json).

Activate from the project root in bash or zsh:

```bash
source scripts/activate.sh
rtk proxy python scripts/check_setup.py
```

The activation script sets absolute JEPA paths and project-local cache locations.
It does not install packages. A nonzero check result means prerequisites remain
missing; see [runtime setup](../notes/setup.md).

## Train the first rover baseline locally

The rover baseline predicts changes in goal distance and angles from current
and previous commands. It learns forward movement, lateral movement, and yaw
(rotation around the vertical axis) in the rover frame. It converts those
predictions back to goal fields. It does not predict the height scan or use
terrain as an input.

With the first HDF5 file at `data/rover/rover_seed_0.hdf5`, run on CPU:

```bash
rtk proxy uv run --no-project --python 3.12 --with torch --with h5py --with numpy python scripts/train_rover_baseline.py
```

The script holds out the last episode by default. For the expanded check, train
on Mars seeds 0 and 1, validate on debug seed 1, and score debug seed 2 separately:

```bash
rtk proxy uv run --no-project --python 3.12 --with torch --with h5py --with numpy python scripts/train_rover_baseline.py \
  --data data/rover/rover_seed_0.hdf5 data/rover/mars-rover-variation-0/rover_seed_1.hdf5 \
  --validation-data data/rover/mars-rover-variation-1/rover_seed_1.hdf5 \
  --test-data data/rover/mars-rover-variation-3/rover_seed_2.hdf5
```

Copy each completed run from `/data/datasets/rover/` on the shared cluster
volume to `data/rover/`, keeping its run directory and `metadata.json`.
The loader rejects duplicate file paths, incompatible dimensions, invalid values,
resets inside episodes, and mismatched action timing. It omits the first
transition after each reset because goal fields can be stale. It also omits the
last action because the following policy observation is not saved. Angle errors
use the shortest distance around the circle. Five-step evaluation stays inside
each episode.

Training first fits 32 transitions, then runs 1,000 optimizer steps for each of
two models: one with commands and one without commands. Training data alone
sets the target scales. No test data enters the optimizer. Each dated output
folder in `runs/rover-baseline/` contains `model.pt`, `no-action.pt`, `metrics.json`,
and a copy of the training source. The new checkpoint schema is `goal-motion-v1`;
older observation-vector checkpoints use a different architecture.

Compare `validation.goal_mse` with `validation.copy_goal_mse` and
`validation_no_action.goal_mse`. The same comparisons are available under `test`
and `test_no_action`. A higher `shuffled_action_goal_mse` shows that correct
commands improve prediction. The `rollout` scores compare five-step predictions
with copying. Scores average the three scaled goal fields, not physical units.

The current seed-2 recording has already been inspected during development.
Treat it as a held-out diagnostic, not a fresh final test. Reserve a future
recording before tuning further. This model assumes a fixed goal within each
episode. It does not establish wheel slip, terrain response, or navigation quality.

To connect the checkpoint to the headless simulator, follow the
[scan-aware navigation guide](../deploy/nautilus-navigation/README.md). It includes
paired comparisons with the original planner and scripted forward motion.

## Finish the reference installation

On a network-enabled, compatible Linux host, follow the downloaded upstream
[installation guide](https://github.com/facebookresearch/jepa-wms#readme), including Conda Python 3.10
and FFmpeg 7. After activating that environment, configure the paths below
without sourcing the Mac virtual environment:

```bash
export JEPAWM_HOME="$PWD/external/jepa-wms"
export JEPAWM_DSET="$PWD/data"
export JEPAWM_LOGS="$PWD/runs"
export JEPAWM_CKPT="$PWD/checkpoints"
rtk proxy uv pip install -e external/jepa-wms
rtk proxy python external/jepa-wms/setup_macros.py
rtk proxy python scripts/check_setup.py
```

The upstream install needs the Metaworld override in
`deploy/nautilus/runtime-setup.sh`. See [results](results.md#reference-planning), then follow the
checkpoint, dataset, MuJoCo, and evaluation instructions. Keep Isaac/RLRoverLab
in a separate environment. Complete both independent milestones in
[the requirements](../REQUIREMENTS.md) before rover model integration.
