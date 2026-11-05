# Setup record

## CPU-only choice

The developer has no NVIDIA GPU. RLRoverLab/Isaac is deferred in favor of the
2D MarsExplorer-Gymnasium prototype. Downloaded its pinned source snapshot
(see `upstream.json`) and created `.venv-rover` with Python 3.10.19.
Installation failed fetching build dependency `setuptools` from PyPI because
DNS resolution failed. The trajectory recorder has not been run against the
environment; only syntax validation is possible until dependencies install.

The root `pyproject.toml` specifies Python >=3.14. It was left unchanged;
the explicit Python 3.10 virtual environments are separate upstream setups.

## Host inspection — 2026-11-03

- Host: macOS / Apple Silicon (`Darwin arm64`); no NVIDIA CUDA platform.
- Available disk at inspection: approximately 82 GiB.
- Git and Git LFS 3.7.1 available.
- uv 0.10.9 available; its default cache is not writable in this session, so installation attempts used `/private/tmp/mars-rover-uv-cache`.
- Conda not found in PATH.
- System Python: 3.9.6. Existing uv-managed Python 3.10.19 used for `.venv-jepa`.
- System FFmpeg: 4.4.4, below the upstream setup's FFmpeg 7 selection.

## Completed

- Created `external/`, `data/`, `checkpoints/`, `runs/`, `notes/`, and `scripts/`.
- Created an isolated `.venv-jepa` using Python 3.10.19.
- Downloaded both upstream source archives at the exact revisions in `upstream.json`.
- Added artifact/cache exclusions in `.gitignore`.
- Added `scripts/activate.sh` for workspace paths and local caches.
- Ran JEPA-WMs `setup_macros.py` successfully after activating the workspace.
- Verified activation selects Python 3.10.19.

The source archives are not Git checkouts. The project root was also not a Git
repository at inspection; no repository initialization or commits were performed.

## Installation attempts and blockers

1. Git clone of both upstreams failed with `Could not resolve host: github.com`.
   HTTPS through curl worked, so commit-pinned GitHub archives were downloaded and extracted instead.
2. `uv pip install --python .venv-jepa/bin/python -e external/jepa-wms`
   failed fetching the upstream Metaworld Git dependency, again due to Git DNS failure.
3. A smaller core dependency installation was attempted and failed resolving
   `pypi.org` while fetching `timm`. No successful ML dependency installation is claimed.

These failures occurred before a usable dependency environment was installed.
Checkpoint loading, dataset inspection, planning, and rover simulation remain unverified.
No dataset or model checkpoint downloads were performed.

## Next execution host

Legacy PointMaze requires its upstream Linux/MuJoCo setup. Isaac/RLRoverLab
requires a supported NVIDIA system in a separate environment. At the downloaded
RLRoverLab revision, its README advertises Isaac Sim 6.0.1, Isaac Lab
v3.0.0-beta2.patch1, and Ubuntu 22.04/24.04; verify the installation guide before
installing those components. Do not install that stack into `.venv-jepa`.

Resolve Git/PyPI access and use the commands in the project README to continue.
Record the final dependency export, hardware, commands, seeds, and run outputs
here after completing installation and evaluation.
