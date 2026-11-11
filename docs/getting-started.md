# Getting started

Use Nautilus for 3D rover simulation and reference JEPA evaluation. Use the
optional CPU prototype for local 2D data experiments. See the
[starter requirements](../REQUIREMENTS.md) for acceptance checks and the
[system overview](architecture.md) for the planned model pipeline.

Run all commands below from the repository root. They use the `rtk` command
proxy. Install RTK first, or remove its prefix when using these tools directly.

## Run on NRP Nautilus

For Step 1 (JEPA checkpoint loading, PointMaze sample inspection, and two planning
episodes), run `rtk proxy bash src/step1/deploy.sh`. See the
[Step 1 guide](../src/step1/README.md) for logs and saved results. The first run
failed on an upstream Metaworld branch reference. Resolve the
[setup blocker](results.md#reference-planning) before retrying.

See [the Nautilus deployment guide](../deploy/nautilus/README.md) for one slim shared
Docker image and separate Kubernetes Jobs for headless RLRoverLab, JEPA checkpoint
loading, dataset preparation, and PointMaze world-model planning. Results are
saved to a shared PVC. Build/publish the image and configure the manifests
before submitting. The A40 rover example is verified: 120 headless environment
steps, exit code 0. Use the deployment guide's `deploy/nautilus-a40` recipe.

## Local workspace

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
missing; see [the setup record](../notes/setup.md).

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

The upstream install currently has the Metaworld dependency blocker described
in [results](results.md#reference-planning). After resolving it, follow the
checkpoint, dataset, MuJoCo, and evaluation instructions. Keep Isaac/RLRoverLab
in a separate environment. Complete both independent milestones in
[the requirements](../REQUIREMENTS.md) before rover model integration.
