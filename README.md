# MarsRover-WM

A physics-informed JEPA world model project for rover navigation. See the
[starter requirements](REQUIREMENTS.md) and [bootstrap](MarsRover_WM_BOOTSTRAP.md).

## Local workspace

**Current hardware choice: CPU-only MarsExplorer on the Mac.** NVIDIA hardware
is not required for this starter path. RLRoverLab/Isaac is deferred.

The lightweight environment has its own Python 3.10 environment, `.venv-rover`,
and a source snapshot at `external/MarsExplorer-Gymnasium`. Install and run with:

```bash
rtk proxy uv pip install --python .venv-rover/bin/python -e external/MarsExplorer-Gymnasium
rtk proxy .venv-rover/bin/python scripts/record_explorer.py
```

Installation currently fails because uv cannot resolve PyPI in this session.
The recording script has been syntax checked but cannot run until dependencies
are installed. Once run, it saves `trajectory.npz`, `metadata.json`, and an SVG
path plot in `runs/explorer/` without opening a graphical window.

MarsExplorer is a 2D, four-action exploration environment. It supports early
data collection and predictive-model experiments; it does not simulate wheel
slip, traction, slopes, or 3D rover dynamics. Its task is coverage rather than
point-goal navigation, so it is a prototype milestone with its own limitations.

The workspace contains commit-pinned JEPA-WMs and RLRoverLab source snapshots
under `external/`, an isolated Python 3.10 environment at `.venv-jepa/`, and
directories for data, checkpoints, and runs. Snapshots have no Git history;
revision metadata is in [notes/upstream.json](notes/upstream.json).

Activate from the project root in bash or zsh:

```bash
source scripts/activate.sh
rtk proxy python scripts/check_setup.py
```

The activation script sets absolute JEPA paths and project-local cache locations.
It does not install packages. A nonzero check result means prerequisites remain
missing; see [the setup record](notes/setup.md).

## Finish the reference installation

On a network-enabled, compatible Linux host, follow the downloaded upstream
[installation guide](external/jepa-wms/README.md), including Conda Python 3.10
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

Next follow the upstream checkpoint, dataset, MuJoCo, and evaluation instructions.
If NVIDIA hardware becomes available, install Isaac/RLRoverLab separately using its upstream guide. Do not integrate
the projects until both independent milestones in `REQUIREMENTS.md` pass.
