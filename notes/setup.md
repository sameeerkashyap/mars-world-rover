# Setup record

## Step 1 status after deployment

Job `mars-jepa-step1` failed during dependency installation. The pinned JEPA
project requests `Farama-Foundation/Metaworld` at branch `master`; Git returned
`fatal: couldn't find remote ref refs/heads/master`. Checkpoint loading, dataset
inspection, and planning did not run. The install log remains under
`/data/runs/setup/mars-jepa-step1-4dg8d/install.log`.
See [results](../docs/results.md) for the current result summary.

## Step 1 deployment — 2026-11-09 UTC

Added `src/step1/` with checkpoint loading, dataset sample inspection, planning,
one bounded job, and a deployment script. Moved the existing JEPA Python entry
points there and updated the Docker build and tests. The script mounts current
source and the shared installer through a ConfigMap, using published image `uv7`.

Deployed Job `mars-jepa-step1` in Nautilus namespace `razvanlab`. Pod
`mars-jepa-step1-4dg8d` was scheduled on `k8s-gpu-1.ucsc.edu`. Setup started and
installed Python 3.10.20 before fetching the pinned JEPA source. This records
deployment, not completed checkpoint loading or planning acceptance.

The job runs the CUDA checkpoint/preprocessor check, PointMaze download, one
dataset sample check, and two seed-0 planning episodes in order. Outputs go to
`/data/runs/jepa/mars-jepa-step1-4dg8d/`; installation logs and package versions
go to `/data/runs/setup/mars-jepa-step1-4dg8d/`. A `SUCCESS` file is written only
after all commands finish successfully. The job has a four-hour deadline and
no automatic retries. The shared PVC remains intact.

Validation: 10 unit tests passed, all three Step 1 Python files parsed, both new
shell scripts passed syntax checks, and the job passed server-side validation.
Full planning results remain pending. See `src/step1/README.md` for log commands.

## Recorded motion results — 2026-11-06 UTC

Job `mars-rover-smoke-video` / pod `mars-rover-smoke-video-6fvkq` completed a
seed-0 physics rollout on the A40 with 120 samples, 24 simulated seconds, no
resets, and 14.46758 m horizontal travel between recorded samples. Results were
copied from the PVC to `runs/rover-video/`. Matplotlib generated a position plot
and a 24-second, 120-frame H.264 top-down replay from the actual CSV data.

RGB camera startup stalled in the SDK rendering pipeline even with the UJITSO
cache disabled. No RGB frames are claimed. An external-process watchdog bounded
the camera attempt, then the worker recorded physics positions without cameras.
The worker was corrected in the live container before the fallback started;
its source and stronger output validation are packaged in the subsequent uv7
image. The recording overlay now defaults to position-only capture to avoid
repeating the renderer stall. The temporary runtime archive attempt was stopped
and its partial file removed; no environment archive remains on CephFS.

## Verified rover run — 2026-11-06 UTC

- Built and pushed `samkash/mars-runtime:uv4`.
- Actual failure captured on the fresh RTX run: pip-installed Kit prompted for
  its EULA and failed with EOF. Added documented `OMNI_KIT_ACCEPT_EULA=YES`
  alongside Docker's `ACCEPT_EULA=Y`, with a regression check.
- Generic GPU capacity was exhausted. Used the A40 Kustomize overlay with one
  CPU and one `nvidia.com/a40`; no reservation tolerations were added.
- Pod `mars-rover-smoke-rz474`, node `bak-hpc1.csub.edu`, completed all 120
  environment steps in `AAURoverEnvSimple-v0` on CUDA. Application exit code 0;
  Job `mars-rover-smoke` Complete. Job 04:21:08–04:26:58 UTC, application
  04:25:17–04:26:55 UTC. Completed pod retained, GPU no longer allocated.
- Image digest: `sha256:6c2d59d37023f45e990aa7f8237c54e2c483cb376b48280bb4da7d892dd75fff`.
- Saved local console to `runs/nautilus-verified/rover-console.log`; PVC copy at
  `/data/runs/rover/mars-rover-smoke-rz474/console.log`.
- Added thread limits, a 15-minute application timeout, and persistent exit-code
  and memory-event diagnostics. The base Job no longer automatically expires.
- Upstream pip metadata conflicts remain reported. The demonstrated result is
  the scripted rover example, not a training or dependency-consistency claim.

## Current-documentation and local preflight review

Re-fetched the live RLRoverLab native guide and GitHub main on 2026-11-05.
The versions and recorded commit are unchanged. Installed actual gdown 6.5.1
and dependencies in isolated local Python 3.12, using PyPI wheels with verified
published SHA256 hashes. Its help and CLI parser ran locally. Its actual asset
download failed resolving Google Drive in this session; no successful download
is claimed.

Pinned gdown 6.5.1, added retries/timeouts and ZIP CRC validation, and added a
regression test using its real parser with only the network transfer mocked.
Added a local Docker preflight using no cluster resources. Updated tag: `uv3`.
Full Docker preflight and GPU rendering remain unverified: this session is
denied Docker socket access and this Mac has no NVIDIA GPU. No additional
cluster jobs were launched.

## Slim-image revision

The requested deployment now uses one small OS-and-uv image,
`samkash/mars-runtime:uv1`, in separate JEPA and rover pods. No upstream source,
Python environment, ML/simulator packages, terrain assets, or weights are baked
into it. Init containers fetch pinned sources and install dependencies into a
pod-local disk-backed `/runtime` volume. The shared PVC contains data/results,
not installed environments. The data-only init installs only Hugging Face tooling.

The image-size gate rejects uncompressed images at or above 1,000,000,000 bytes.
Actual image size and native Isaac runtime remain unverified because Docker
socket access is denied in this session. Small pulls trade against slower cold
starts and repeated multi-GB runtime downloads. Previous two-image Dockerfiles
were removed and their image tags are no longer referenced by Jobs.

Validation for this revision: all five manifests pass strict Kubernetes schema
validation, all five unit tests pass, and all four shell scripts pass syntax
checks. The image-size test covers the 1 GB boundary; it is not a measurement
of the unbuilt image.

## Nautilus deployment preparation — 2026-11-05

Retry after the developer started Docker/Kubernetes: Docker socket connection
failed with `connect: operation not permitted`; Nautilus API connection to
`67.58.53.148:443` failed with the same permission error. These are session
sandbox access failures, not evidence that either service is stopped. Image
builds, pushes, and cluster submissions remain unperformed.

Added `docker/` image recipes and `deploy/nautilus/` finite Jobs for the pinned
RLRoverLab/Isaac stack, JEPA model loading, CPU-only PointMaze data preparation,
and a reduced-budget upstream PointMaze planning evaluation. A shared PVC stores
datasets, checkpoints, logs, and configuration. Namespace `razvanlab` and Docker
Hub account `samkash` were confirmed by the developer and configured in all manifests.

The local kubeconfig selects `nautilus` / `razvanlab`. Cluster inspection failed
at OIDC authentication due to DNS resolution of `authentik.nrp-nautilus.io`.
`docker info` reported that the Docker daemon is not running. No images were
built/pushed and no cluster objects were created.

Local validation: three unittest checks passed for finite, resource-bounded,
persistently logged Jobs; RTX/headless scheduling; and the modified planning
config. Both Bash launchers passed `bash -n`, and all new Python files parsed.
PyYAML 6.0.3 source was fetched through curl into `/private/tmp` for validation
without changing project dependencies. All five manifests also passed strict kubeconform validation against Kubernetes
1.34 schemas. Live admission checks and all GPU/container runtime checks remain
outstanding.

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
