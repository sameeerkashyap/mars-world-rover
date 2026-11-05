# MarsRover-WM — Project Bootstrap

**Goal:** Build a physics-informed, action-conditioned JEPA world model that predicts rover–terrain interactions and supports autonomous goal-directed navigation on unfamiliar Martian terrain.

**First milestone:** Reproduce an existing JEPA-WM planning task and independently run a rover environment. **Do not integrate them until both run separately.**

## 1. Reference stack

| Role | Starting resource | Why use it |
| --- | --- | --- |
| World model and planner | [Meta JEPA-WMs](https://github.com/facebookresearch/jepa-wms) | Reference encoder, action-conditioned latent predictor, training, pretrained checkpoints, and planning evaluations |
| Initial navigation reference | JEPA-WMs **PointMaze** task | Verify end-to-end latent planning before adding a rover |
| Rover simulator (target integration) | [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab) | Isaac Sim / Isaac Lab rover navigation environments with state, height-map, camera, and RGB-D observations |
| Lightweight optional prototype | [MarsExplorer-Gymnasium](https://github.com/markub3327/MarsExplorer-Gymnasium) | Fast Gymnasium exploration baseline when Isaac setup is impractical |
| Later Mars-specific evaluation | [MarsLab](https://kimhoyun-robotair.github.io/MarsLab/) | More Mars-specific rover terrain and sensor simulation; not a day-one dependency |

**Recommended sequence:** JEPA-WMs → PointMaze reproduction → RLRoverLab running separately → collect rover trajectories → rover JEPA-WM → physics-informed loss → MPC navigation → Mars-specific terrain evaluation.

## 2. Prerequisites

### Required before starting

- **Git**, Git LFS if a selected dataset/checkpoint requires it, and sufficient disk space for model weights, trajectory data, and logs.
- **Conda/Miniconda** and **uv** for the JEPA-WMs environment.
- **Python 3.10** for the JEPA-WMs environment, following its upstream installation guide.
- **PyTorch** and related dependencies installed through the upstream JEPA-WMs project; verify GPU support in your environment.
- A **CUDA-capable NVIDIA GPU is strongly recommended** for model training and especially for later Isaac-based simulation. CPU-only development can still cover configuration, preprocessing, and lightweight tests.
- Familiarity with **PyTorch**, **Gymnasium-style** `reset/step` APIs, sequence datasets, and model-predictive control (MPC).

### Additional prerequisites when moving to RLRoverLab

- A **compatible NVIDIA Isaac Sim + Isaac Lab installation**, ideally on a supported Linux / NVIDIA GPU configuration.
- Follow the [RLRoverLab installation guide](https://abmorobotics.github.io/RLRoverLab/) for its currently supported Isaac versions, drivers, Docker/native options, and task commands. **Do not assume RLRoverLab and JEPA-WMs should share one Python environment.**
- Enough compute for 3D simulation and the ability to record observations, commands, and ground-truth trajectories.

### Compatibility warning

JEPA-WMs **PointMaze** evaluation uses legacy **MuJoCo 2.1 / `mujoco-py`** via `d4rl`. Its upstream README documents a Linux-specific setup. On a Mac, use Linux (local or remote) for this evaluation, or begin by loading the pretrained model and reproducing a non-PointMaze example. Keep a record of exact versions and commit hashes.

## 3. Create the workspace

```bash
mkdir -p marsrover-wm/{external,data,checkpoints,runs,notes}
cd marsrover-wm

git clone https://github.com/facebookresearch/jepa-wms.git external/jepa-wms
git clone https://github.com/abmoRobotics/RLRoverLab.git external/RLRoverLab

# Optional, only if choosing the lightweight Gymnasium route:
git clone https://github.com/markub3327/MarsExplorer-Gymnasium.git external/MarsExplorer-Gymnasium
```

Keep third-party repositories under `external/`; implement your own integration and physics code in the project root rather than editing all upstream sources directly. Record the commit ID of every upstream repository used.

## 4. Install and verify JEPA-WMs

Run these commands from the `marsrover-wm` workspace:

```bash
# Install uv if needed (official installer):
curl -LsSf https://astral.sh/uv/install.sh | sh

conda create -n jepa-wms python=3.10 ffmpeg=7 -c conda-forge -y
conda activate jepa-wms
cd external/jepa-wms
uv pip install -e .
python -c "import torchcodec; print('torchcodec OK')"
```

Configure the environment variables documented in the [upstream JEPA-WMs README](https://github.com/facebookresearch/jepa-wms), particularly `JEPAWM_HOME`, `JEPAWM_DSET`, `JEPAWM_LOGS`, and optionally `JEPAWM_CKPT`. Run the repository's `setup_macros.py` as instructed there; use real absolute paths for your workspace.

**First sanity check:** Load the PointMaze checkpoint using the repository's documented Torch Hub entry point:

```python
import torch
model, preprocessor = torch.hub.load(
    "facebookresearch/jepa-wms",
    "jepa_wm_pointmaze"
)
print(type(model))
```

Loading the checkpoint **does not** mean navigation evaluation is working. To reproduce PointMaze planning, follow the upstream PointMaze dataset, legacy MuJoCo, and planning-evaluation instructions. The repository lists the PointMaze dataset and its download helper, including:

```bash
# From external/jepa-wms, with environment paths configured:
python src/scripts/download_data.py --list
python src/scripts/download_data.py --dataset pointmaze
```

**Exit condition:** You can load a pretrained checkpoint, inspect a PointMaze sample, and run at least one upstream planning evaluation with reproducible logs.

## 5. Run a rover environment independently

Open [RLRoverLab](https://github.com/abmoRobotics/RLRoverLab) and follow its **official installation guide and examples**. Start with a single provided rover navigation task. Do not yet connect JEPA-WMs or train an agent.

Confirm that the environment exposes or can log:

- Observation: camera or local height map, plus permitted onboard rover-state measurements.
- Action: steering / angular-velocity and drive / velocity commands, as supported by the chosen environment.
- Transition: next observation and pose after a known simulation time step.
- Training diagnostics: contact/collision, slope, slip or wheel speed (if available); clearly distinguish simulator privileged state from agent observations.
- Episode signals: goal, termination, and success/failure.

**Exit condition:** The rover moves under scripted/random commands, and one complete time-aligned trajectory is saved and visualized. If the simulator does not provide wheel-slip signals, implement that diagnostic later rather than treating an unavailable field as present.

## 6. Reference architecture (target, not implemented yet)

```text
RLRoverLab simulator
  └── (observation_t, rover_state_t, action_t, observation_t+1, ...)
           │
           ▼
    Rover trajectory dataset
           │
           ├── context encoder Eθ(observation_t, state_t) → z_t
           │
           ├── target encoder Ē(observation_t+1, state_t+1) → z_target
           │
           ▼
    action-conditioned latent predictor Fφ(z_t, action_t) → ẑ_t+1
           │
           ├── JEPA latent prediction loss
           ├── physical-state head → displacement, heading, slip estimate
           └── physics consistency loss → rover motion / terrain constraints
           │
           ▼
    MPC planner (sample action sequences → imagined rollouts)
           │
           ▼
    Execute first action → observe → replan
```

**Architecture objective:** Preserve JEPA-WMs' latent predictive learning and planning pattern while adapting observations and action semantics to rover navigation. Introduce a **physically interpretable prediction head** before applying physical losses; arbitrary embedding dimensions are not physical coordinates. Physics consistency should permit genuine wheel slip rather than wrongly enforce no-slip motion.

**Initial scope:** Goal-directed navigation with collision avoidance and terrain-dependent traction. **Out of scope for bootstrap:** Full rover autonomy, detailed terramechanics, foundation-model pretraining, RL actor–critic training, and integration with atmospheric MarsWorld.

## 7. Roadmap and completion gates

| Phase | Focus | Completion gate |
| --- | --- | --- |
| **0 — Reference reproduction** | Install JEPA-WMs, load checkpoint, reproduce one planning run | Working reference + logs |
| **1 — Rover simulator** | Install RLRoverLab separately; run scripted navigation | Recorded rover episodes |
| **2 — Data adapter** | Normalize observations/actions/time steps; build sequence loader | Train-ready rover transition batches |
| **3 — Rover JEPA-WM** | Adapt encoder/predictor and retrain rover dynamics | Validated one-step and multi-step latent/state predictions |
| **4 — Physics-informed model** | Add state head and rover motion/traction constraints | Comparable model with and without physics loss |
| **5 — Navigation planner** | Adapt MPC costs for goal progress, collisions, and traction risks | Closed-loop autonomous goal reaching |
| **6 — Mars evaluation** | Add held-out terrain maps, friction regimes, optionally MarsLab | Reproducible navigation comparisons on unseen terrain |

**Priority:** Complete phases 0 and 1 before writing the rover-specific model. Move on only once each completion gate is met.

## 8. First-session checklist

- [ ] Create workspace and clone JEPA-WMs and RLRoverLab.
- [ ] Establish the JEPA-WMs Python environment and verify imports.
- [ ] Load the pretrained PointMaze model.
- [ ] Read the upstream PointMaze planning/evaluation instructions and identify its simulator requirements.
- [ ] Read RLRoverLab's installation guide, choose a supported rover navigation task, and verify hardware compatibility.
- [ ] Save the dependency versions and upstream commit IDs in `notes/setup.md`.

## Source READMEs

- JEPA-WMs: https://github.com/facebookresearch/jepa-wms
- RLRoverLab: https://github.com/abmoRobotics/RLRoverLab
- RLRoverLab docs: https://abmorobotics.github.io/RLRoverLab/
- Optional lightweight rover: https://github.com/markub3327/MarsExplorer-Gymnasium
- Later MarsLab: https://kimhoyun-robotair.github.io/MarsLab/

> This bootstrap describes a proposed integration of independently maintained projects. It is **not** a tested, plug-and-play JEPA-WMs/RLRoverLab installation; the integration adapter, rover training data, physics head, and final controller remain project work.
