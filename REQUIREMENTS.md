# MarsRover-WM — Starter Requirements

Source: [Project bootstrap](MarsRover_WM_BOOTSTRAP.md).

## Active hardware decision

The remote execution target is now **NRP Nautilus Kubernetes**. See
[the deployment guide](deploy/nautilus/README.md). Run RLRoverLab headlessly on
compatible RTX nodes and JEPA in a separate CUDA image/Job. Their independent
acceptance gates still apply; deployment files alone do not establish a working
simulation or planning result. The CPU prototype below remains the local fallback.

The developer has no NVIDIA GPU. Use CPU-only **MarsExplorer-Gymnasium** for
the initial environment and trajectory-collection prototype. Defer
RLRoverLab/Isaac and its GPU requirements. Keep JEPA-WMs reproduction separate;
legacy PointMaze still needs a compatible Linux setup even without Isaac.

For this prototype, Milestone B means running one complete exploration episode
and saving aligned observations, actions, rewards, termination flags, step
indices, and a path visualization. Record grid positions as privileged diagnostics.
Physical timing, camera/height-map sensors, point goals, slip, and traction are
unavailable in this substitute and must not be claimed as validated. The original
3D rover acceptance criteria below remain deferred. See the README for commands.

## Project objective

Build a physics-informed, action-conditioned JEPA world model for rover–terrain prediction and goal-directed navigation on unfamiliar Martian terrain.

The first milestone is two independent demonstrations:

1. Reproduce a JEPA-WMs planning evaluation, preferably PointMaze.
2. Run a rover simulator and save one complete, time-aligned trajectory.

Do not begin rover model integration until both demonstrations pass their acceptance checks.

## Development prerequisites

| Requirement | Needed for | Verification |
| --- | --- | --- |
| Git | Fetch and track upstream projects | Record each upstream URL and commit hash |
| Git LFS, if required by selected assets | Download datasets or checkpoints | Confirm downloaded assets are actual files, not LFS pointers |
| Conda or Miniconda, plus uv | Isolated JEPA-WMs environment | Both tools are available in the development shell |
| Python 3.10 | JEPA-WMs | Confirm the active environment's Python version |
| FFmpeg 7 and upstream PyTorch dependencies | JEPA-WMs data/model stack | Follow upstream installation instructions; verify `torch` and `torchcodec` imports |
| Dataset, checkpoint, and log storage | Reproduction and trajectory recording | Record available disk space and chosen storage paths before downloads |
| Compatible NVIDIA GPU and drivers | Recommended for training; Isaac simulation compatibility must be checked | Record GPU, driver, and CUDA support against the chosen upstream versions |
| Supported Linux environment | Legacy PointMaze evaluation and preferred Isaac setup | Confirm compatibility before installation |

CPU-only development can cover documentation, configuration, preprocessing, and lightweight checks. The bootstrap specifies no fixed GPU memory or storage minimum; determine capacity from the selected simulator, checkpoint, and dataset.

PointMaze uses legacy MuJoCo 2.1 / `mujoco-py` through `d4rl`, with Linux-specific setup. On macOS, use a compatible Linux machine for that evaluation. Loading a checkpoint on macOS is a useful preliminary check but does not satisfy the planning milestone. If selecting another upstream planning task, document the choice and its acceptance evidence.

## Workspace and environment requirements

Use the current repository as the project root:

```text
external/          # Independent upstream checkouts
  jepa-wms/
  RLRoverLab/
data/              # Downloaded datasets and recorded rover trajectories
checkpoints/       # Pretrained and locally trained weights
runs/              # Evaluation logs, metrics, and visualizations
notes/
  setup.md         # Environment details, upstream commits, commands, results
```

- Keep JEPA-WMs and Isaac/RLRoverLab in separate environments.
- Install dependencies using each project's supported installation guide. Do not assume compatible Python or package versions across the two stacks.
- Configure absolute paths for `JEPAWM_HOME`, `JEPAWM_DSET`, `JEPAWM_LOGS`, and, when used, `JEPAWM_CKPT`; run the upstream macro setup as documented.
- Keep project-specific adapters and physics code outside upstream checkouts.
- Exclude downloaded datasets, checkpoints, generated runs, and local environments from version control when setting up the repository.
- Record exact versions and commit hashes after installation. A shared pinned Python requirements file should wait until the two environments have been verified independently.

## Milestone A: Reference planning

Required behavior:

- Install JEPA-WMs and verify its required imports.
- Load the pretrained PointMaze checkpoint and preprocessor.
- Download the selected evaluation dataset and inspect at least one sample.
- Install the selected task's simulator dependencies.
- Execute at least one upstream planning evaluation.

Acceptance evidence:

- A saved evaluation command and configuration, including the seed when supported.
- Upstream commit, dependency versions, dataset/checkpoint identifiers, and hardware details in `notes/setup.md`.
- Saved planning logs and reported outcome metrics under `runs/`.

Checkpoint loading alone does not pass this milestone. No numerical performance threshold is specified yet; preserve the upstream task's evaluation metrics for comparison.

## Milestone B: Independent rover simulation

Use [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab) with a compatible Isaac Sim / Isaac Lab installation, following its [installation guide](https://abmorobotics.github.io/RLRoverLab/).

Required behavior:

- Launch one supported rover navigation task.
- Reset the environment and move the rover with scripted or random commands.
- Record and visualize one complete episode without connecting JEPA-WMs.
- Document the task, observation fields, action meanings, units, coordinate frames, and simulation/control time steps.

Minimum recorded trajectory information:

| Information | Requirement |
| --- | --- |
| Episode and step identifiers | Preserve ordering and episode boundaries |
| Time | Timestamp or a documented fixed transition interval |
| Observations | Camera or local height map, plus available onboard rover-state measurements |
| Actions | Actual commands applied, with units and valid ranges |
| Next observations and pose | Align with the action and transition interval |
| Goal and episode outcome | Record termination and success/failure signals |
| Simulator diagnostics | Record available collision/contact, slope, wheel speed, or slip signals; explicitly mark unavailable fields |

Keep simulator privileged state separate from inputs available to the navigation agent. Do not assume wheel-slip measurements exist.

Acceptance evidence:

- A saved trajectory with a documented field schema and no unexplained timing gaps.
- A plot or video showing rover motion from that trajectory.
- A reproducible launch/recording command and environment details in `notes/setup.md`.

If Isaac setup is impractical, evaluate [MarsExplorer-Gymnasium](https://github.com/markub3327/MarsExplorer-Gymnasium) as an optional prototype. Record that substitution and its limitations; it does not establish RLRoverLab compatibility.

## Requirements after the starter milestones

These are future deliverables, not prerequisites for the first working demonstrations:

1. A trajectory adapter that normalizes observations, actions, and time steps into sequence batches.
2. A rover encoder and action-conditioned latent predictor, evaluated on one-step and multi-step predictions.
3. An interpretable physical-state prediction head before adding physics losses. Latent embedding dimensions must not be treated as physical coordinates.
4. Physics constraints that permit real wheel slip, with an otherwise comparable baseline without physics loss.
5. An MPC controller evaluated on goal progress, collisions, and terrain-dependent traction risks.
6. Reproducible comparisons on held-out terrain and friction conditions; MarsLab is optional later work.

Bootstrap scope excludes full rover autonomy, detailed terramechanics, foundation-model pretraining, RL actor–critic training, and atmospheric MarsWorld integration.

## Getting-started checklist

- [ ] Choose a compatible Linux/GPU execution host and record its capabilities.
- [ ] Create the workspace directories and ignore generated/downloaded artifacts.
- [ ] Clone JEPA-WMs and RLRoverLab; record their commit hashes.
- [ ] Create and verify the JEPA-WMs environment.
- [ ] Configure storage paths, load the checkpoint, and inspect a dataset sample.
- [ ] Complete Milestone A and save its evidence.
- [ ] Install the separate rover environment and select a supported task.
- [ ] Complete Milestone B and save its evidence.
- [ ] Review both acceptance results before starting the rover data adapter.

This document defines the starter requirements; it does not certify that dependencies, simulators, or evaluations have been installed or run.
