# Nautilus runtime installation

## Verified headless example

Recorded position results are available in `runs/rover-video/`: `trajectory.png`,
`trajectory-replay.mp4`, `trajectory.csv`, and `summary.json`. The recorded run
completed 120 steps / 24 simulated seconds with no resets and 14.47 m of
horizontal travel between recorded samples. The MP4 is a top-down animation of
measured world positions, **not an RGB camera recording**. The camera-enabled
Isaac startup stalled; its bounded fallback successfully recorded physics state.

To reproduce the position recording, use `kubectl apply -k deploy/nautilus-video`
after publishing `samkash/mars-runtime:uv7`. This recipe skips the unreliable
camera startup using `--no-video` and saves the CSV/JSON on the shared PVC under
`/data/runs/rover-video/`. A temporary, read-only CPU pod for copying artifacts
is provided in `artifacts-pod.yaml`; remove it after retrieval. The source keeps
camera capture as an experimental option with an external-process timeout.

On 2026-11-06 UTC, Job `mars-rover-smoke` completed successfully using the A40
overlay. Pod `mars-rover-smoke-rz474` on `bak-hpc1.csub.edu` ran
`AAURoverEnvSimple-v0` with one rover for **120 environment steps** and exited
with code **0**. Total Job time, including runtime installation: 5m50s.
Application execution: 98 seconds. Resources: one A40, one CPU, 16 GiB app RAM,
8 GiB installer RAM, and 100 GiB ephemeral storage.

The actual startup blocker was the missing pip-runtime EULA variable, not the
dependency metadata warnings. `OMNI_KIT_ACCEPT_EULA=YES` fixes the prompt/EOF
failure. The installer still reports upstream dependency metadata conflicts;
this verifies the scripted demo, not training or complete package consistency.

The working image is `samkash/mars-runtime:uv4`, digest
`sha256:6c2d59d37023f45e990aa7f8237c54e2c483cb376b48280bb4da7d892dd75fff`.
Local console copy: `runs/nautilus-verified/rover-console.log`.
Persistent console: `/data/runs/rover/mars-rover-smoke-rz474/console.log`.
The completed pod is retained for inspection; its GPU has been released.

To repeat the verified run:

```bash
rtk kubectl --context nautilus -n razvanlab delete job mars-rover-smoke --ignore-not-found --wait=true
rtk kubectl --context nautilus -n razvanlab apply -k deploy/nautilus-a40
rtk kubectl --context nautilus -n razvanlab logs -f job/mars-rover-smoke -c setup
rtk kubectl --context nautilus -n razvanlab wait --for=condition=complete --timeout=14400s job/mars-rover-smoke
rtk kubectl --context nautilus -n razvanlab logs job/mars-rover-smoke -c rover
```

One small image, **separate JEPA and rover pods**. The image contains Ubuntu,
uv, Git, compilers, and shared OS libraries. It contains no Python environment,
PyTorch, Isaac, upstream checkouts, weights, or terrain assets.

Git LFS is downloaded as a pinned Linux AMD64 binary during rover setup. It is
needed by Isaac Lab; it is not installed through apt in the image, avoiding the
reported Ubuntu `git-lfs` package configuration failure during the image build.

| Job | Init-container setup | Application container |
| --- | --- | --- |
| `rover-job.yaml` | Python 3.12, Isaac Sim 6.0.1.0, pinned Isaac Lab and RLRoverLab, terrain assets | 120 headless rover steps on RTX |
| `jepa-job.yaml` | Python 3.10, pinned JEPA-WMs, FFmpeg 7, MuJoCo 2.1 | CUDA checkpoint loading check |
| `jepa-data-job.yaml` | Python 3.10, pinned JEPA source, Hugging Face download tooling only | CPU PointMaze dataset download |
| `jepa-plan-job.yaml` | Same JEPA setup | Two PointMaze planning episodes on CUDA |

Each pod gets its own `/runtime` disk-backed `emptyDir` and virtual environment.
Init and application containers share that pod's scratch volume. JEPA and Isaac
are never installed into the same environment. Only datasets, weights, and
results are shared through the `mars-world-data` PVC in namespace `razvanlab`.

## Size and startup tradeoff

The developer accepted a small increase above the original 1 GB target.
The reported build is 1,094,245,367 bytes; the guard now allows up to 1.2 GB.
The **published image must be below 1,200,000,000 bytes uncompressed**. The build
commands below enforce that before pushing; the size is not yet measured here
because this session cannot connect to Docker's socket. A small image does not
make the installed Isaac stack less than 1 GB. The supported native Isaac
installation downloads many GB at startup, and new pods repeat the installation.

Rover pods request 100 GiB ephemeral storage (90 GiB runtime volume); JEPA pods
request 40 GiB (35 GiB runtime volume). Package installations and caches stay on
node-local scratch, never on CephFS. uv's cache is cleaned after installation.
The downloader uses only 5 GiB ephemeral storage and installs no Torch/Isaac.

Init containers use CPU, but a GPU pod's GPU is still reserved during its setup.
The four-hour Job deadline includes installation and execution. This trades
smaller, faster image pulls for slower cold starts; it does not promise faster
end-to-end startup. A failed init never starts the GPU application.

## Build, measure, and push

Run these on an AMD64 Docker builder, or use Docker Desktop's AMD64 emulation
for the small OS-only build. No NVIDIA registry login, GPU, local upstream
checkout, or model download is required to build this image.

```bash
rtk proxy docker build --platform linux/amd64 -f docker/runtime.Dockerfile -t samkash/mars-runtime:uv3 .
rtk proxy bash docker/check-image-size.sh samkash/mars-runtime:uv3
# Validate installation before publishing or requesting cluster GPUs.
rtk proxy bash docker/local-preflight.sh rover
rtk proxy bash docker/local-preflight.sh jepa
rtk proxy bash docker/local-preflight.sh data
rtk proxy docker login
rtk proxy docker push samkash/mars-runtime:uv3
```

The size check fails at or above 1.2 GB. Do not push a failing image. Use an image
digest in the manifests after publishing when possible. The former large
`mars-rover` and `mars-jepa` images and the upstream Isaac base build are no
longer used by these Jobs.

The local preflight runs the exact init installer inside the built Linux image,
without requesting a GPU or creating Kubernetes resources. It checks package
consistency; rover versions and terrain archive CRC/extraction; and JEPA's
TorchCodec/MuJoCo imports and CPU checkpoint loading. Reports persist in
`runs/preflight/<mode>/runs/setup/<container>/`. A failure exits nonzero; resolve
it before deploying. Allow at least 50 GB of free Docker VM disk for rover
installation, preferably the pod's 100 GiB budget. AMD64 emulation on Apple
Silicon can be slow or expose emulation-specific failures.

This does not test RTX rendering, GPU physics, or actual rover navigation on
the Mac. Those need a final smoke test on supported Linux/NVIDIA hardware.
The terrain downloader is pinned to gdown 6.5.1, uses a raw file ID, and has
timeouts/retries. A real CLI parser regression test covers those arguments.
Upstream's native guide still calls an older downloader helper; this is an
explicit compatibility fix for its removed `--fuzzy` argument.

## Check access and scheduling

```bash
rtk proxy kubectl --context nautilus -n razvanlab auth can-i create jobs
rtk proxy kubectl --context nautilus -n razvanlab auth can-i create persistentvolumeclaims
rtk proxy kubectl --context nautilus get nodes -L nvidia.com/gpu.product,nvidia.com/cuda.driver.major,nvidia.com/cuda.driver.minor
rtk proxy kubectl --context nautilus get storageclasses
rtk proxy kubectl --context nautilus -n razvanlab get resourcequota
```

The one-rover smoke test allows more RTX product labels: 3090/Ti, 4080/SUPER,
4090, A4000/A4500/A5000, 5000/6000 Ada, Quadro RTX 5000/6000, and TITAN RTX.
These candidates have RT cores and at least 16 GB VRAM; this does not establish
Isaac 6 compatibility on older cards. Exact cluster labels must match.
The application now requests/limits 4 CPUs and 16 GiB RAM to ease scheduling.
This is experimental sizing below the listed 32 GB host requirement: if it is
OOMKilled, increase RAM before retrying. The 100 GiB disk request remains for
runtime installation. A40/A6000/RTX8000 require dedicated Nautilus GPU resource
keys and are not enabled by this generic `nvidia.com/gpu` Job.
Verify exact labels and Isaac
6.0.1 driver compatibility before submitting. Isaac 6 requirements list 32 GB
host RAM, 16 GB VRAM, and tested Linux driver 580.95.05. RTX A4000 is below the
listed GPU minimum. A100/H100 lack RT cores and are not supported by this Isaac
renderer even headless. No display, X11 forwarding, privileged container,
host networking, or host driver installation is configured.

For JEPA on A100, replace both `nvidia.com/gpu` resource keys with
`nvidia.com/a100` and set `priorityClassName: opportunistic` under the pod spec
unless your group has an approved reservation/quota. Opportunistic runs can be
preempted. GH200 requires ARM64; this image/workflow targets AMD64.

The PVC initially requests 100 GiB ReadWriteMany on `rook-cephfs`; select an
appropriate available storage class/region before its creation. All containers
set equal CPU, memory, and ephemeral-storage requests and limits. Measure usage
and adjust subsequent runs. Private Docker Hub images need an image pull Secret
referenced by both application and init containers via pod `imagePullSecrets`.
Keep credentials out of manifests and source control.

The rover uses both `ACCEPT_EULA=Y` and `OMNI_KIT_ACCEPT_EULA=YES`. The latter
is required by the pip-installed Kit runtime; without it the example prompts
for input and exits with `Unable to bootstrap inner kit kernel: EOF when reading
a line`. These settings accept the NVIDIA Isaac Sim EULA. NRP access must comply
with your account's accepted AUP.

The rover smoke image is now `samkash/mars-runtime:uv4`. It limits BLAS/OpenMP
threads to one, captures exit codes and memory events, and bounds the example
process to 15 minutes. Increase that bound before using this launcher for long
training. The base Job has no automatic finished-job TTL so failure status can
be inspected before manual cleanup.

If the generic GPU pool has no capacity, use the A40 overlay. A40 has RT cores
and is advertised through a dedicated `nvidia.com/a40` resource. This overlay
requests one CPU for both installation and the one-rover smoke application:

```bash
rtk proxy kubectl --context nautilus -n razvanlab delete job mars-rover-smoke --ignore-not-found --wait=true
rtk proxy kubectl --context nautilus -n razvanlab apply -k deploy/nautilus-a40
```

This replaces the existing Job and its temporary runtime. It does not delete
the shared PVC. The base recipe remains available with `-f rover-job.yaml` or
`-k deploy/nautilus` for the generic GPU pool, including A10 and L4/L40/L40S.

## Submit sequentially

Docker and Nautilus access are working, and the A40 example above is verified.
JEPA Jobs have not been run on the cluster in this session.
Existing completed Jobs must be removed or renamed
before changing their immutable pod templates; do not delete running work or PVCs.

```bash
rtk proxy kubectl --context nautilus -n razvanlab apply --dry-run=server -f deploy/nautilus/storage.yaml
rtk proxy kubectl --context nautilus -n razvanlab apply -f deploy/nautilus/storage.yaml
rtk proxy kubectl --context nautilus -n razvanlab apply --dry-run=server -f deploy/nautilus/rover-job.yaml
rtk proxy kubectl --context nautilus -n razvanlab apply -f deploy/nautilus/rover-job.yaml
rtk proxy kubectl --context nautilus -n razvanlab logs -f job/mars-rover-smoke -c setup
rtk proxy kubectl --context nautilus -n razvanlab wait --for=condition=complete --timeout=14400s job/mars-rover-smoke
rtk proxy kubectl --context nautilus -n razvanlab logs job/mars-rover-smoke -c rover
```

If setup has already finished, its logs remain available with `-c setup`. If
Kubernetes reports the application is still initializing, follow the setup logs
first. Describe the pod for scheduling, pull, OOM, or disk-eviction errors.

Then submit the remaining Jobs **one at a time**, waiting for completion before
the next. Stage the dataset before planning:

```bash
rtk proxy kubectl --context nautilus -n razvanlab apply -f deploy/nautilus/jepa-job.yaml
rtk proxy kubectl --context nautilus -n razvanlab wait --for=condition=complete --timeout=14400s job/mars-jepa-smoke
rtk proxy kubectl --context nautilus -n razvanlab apply -f deploy/nautilus/jepa-data-job.yaml
rtk proxy kubectl --context nautilus -n razvanlab wait --for=condition=complete --timeout=14400s job/mars-jepa-data
rtk proxy kubectl --context nautilus -n razvanlab apply -f deploy/nautilus/jepa-plan-job.yaml
rtk proxy kubectl --context nautilus -n razvanlab wait --for=condition=complete --timeout=14400s job/mars-jepa-pointmaze
```

## Reproducibility, outputs, and longer runs

Setup shallow-fetches and verifies the source commits in `notes/upstream.json`.
Isaac Lab is fixed to `ffff603eafc6b74264a5261cc0183d6a65390d78`.
Python packages are installed with uv, except where the upstream Isaac Lab
installer manages its own dependencies. Runtime resolutions are not fully locked;
setup saves `packages.txt`, source revisions, and `install.log` under
`/data/runs/setup/<pod>/`. Network/PyPI/NVIDIA/Google Drive availability and asset
quotas are runtime dependencies. The supported native install retains Isaac's
extension cache extras to avoid further extension downloads during simulation.

JEPA uses FFmpeg 7 shared libraries from conda-forge via runtime micromamba,
TorchCodec 0.3 for the upstream Torch 2.7 / torchvision 0.22 family, and legacy
MuJoCo 2.1. The planning check compiles/imports `mujoco_py`, saves its generated
config, and uses a reduced CEM budget: 3 iterations and 32 samples, with optional
decoder heads disabled. This is a functional evaluation, not a paper benchmark.

Outputs survive pod deletion:

```text
/data/datasets/              PointMaze dataset
/data/checkpoints/           Persistent weights and future training checkpoints
/data/cache/                 Model download caches, not installed packages
/data/runs/setup/<pod>/      Installation log, package versions, source pins
/data/runs/rover/<pod>/      Console, arguments, stack verification, SUCCESS
/data/runs/rover/training/   Optional upstream RL training logs/checkpoints
/data/runs/jepa/<pod>/       Model report or planning config/results, SUCCESS
```

The initial rover demo does not save a training trajectory or connect to JEPA.
Keep both independent reference checks before developing the rover data adapter.
For bounded upstream RL training, change the rover application arguments to:

```yaml
args: [examples/02_training/train.py, --task, AAURoverEnvSimple-v0, --num_envs, "16", --steps, "10000", --viz, none]
```

For JEPA training, use application arguments
`[python, -m, app.main, --fname, /data/configs/train.yaml, --debug]` with a real
validated config and dataset. There is no rover-trained JEPA checkpoint yet.

Download PVC results through your existing Coder/Jupyter workspace or a finite
CPU data-transfer task. Do not create idle GPU Jobs. Finished Jobs/pods expire
after one day; the PVC remains. Remove unused Jobs after review. Delete the PVC
only after copying all needed results elsewhere, because deletion can destroy data.

## Local checks

With source snapshots available for the planning-config test:

```bash
rtk proxy uv run --no-project --with pyyaml --with gdown==6.5.1 python -m unittest discover -s tests -v
rtk proxy bash -n deploy/nautilus/runtime-setup.sh
rtk proxy bash -n deploy/nautilus/rover-run.sh
rtk proxy bash -n deploy/nautilus/jepa-run.sh
rtk proxy bash -n docker/check-image-size.sh
```

Local tests cover bounded Jobs, isolated runtime volumes, headless RTX selection,
planning configuration, and the 1.2 GB size gate. They do not exercise native Isaac,
package downloads, GPU drivers, cluster admission, or measure the actual image.

Documentation checked on 2026-11-05: the published native guide still specifies
Python 3.12, Isaac Sim 6.0.1.0, Isaac Lab v3.0.0-beta2.patch1 at `ffff603`, and
`--viz none`. GitHub main still resolves to the recorded RLRoverLab `578906ad`
commit. Docker is upstream's validated installation path; our small bootstrap
image follows its native installation commands instead of shipping that base.

Sources: [RLRoverLab native installation](https://abmorobotics.github.io/RLRoverLab/installation/native.html),
[Docker guide](https://abmorobotics.github.io/RLRoverLab/installation/docker.html),
[Isaac requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/installation/requirements.html),
[NRP Jobs](https://nrp.ai/documentation/userdocs/running/jobs/),
[NRP storage](https://nrp.ai/documentation/userdocs/storage/ceph/).
