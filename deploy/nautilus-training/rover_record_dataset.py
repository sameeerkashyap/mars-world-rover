"""Record a small action-conditioned rover dataset with RLRoverLab's recorder."""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AAURoverEnvSimple-v0")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--terrain", choices=("mars", "debug"), default="mars")
parser.add_argument("--sweep-index", type=int)
parser.add_argument("--varied", action="store_true", help="Record forward, reverse, turn, and stop commands")
parser.add_argument("--output", type=Path, default=Path("/data/datasets/rover"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1:
    parser.error("--steps must be positive")
if args.sweep_index is not None:
    if not 0 <= args.sweep_index < 6:
        parser.error("--sweep-index must be between 0 and 5")
    args.seed = 1 + args.sweep_index // 2
    args.terrain = ("mars", "debug")[args.sweep_index % 2]
args.varied = args.varied or args.sweep_index is not None
args.enable_cameras = False
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
import rover_envs.envs.navigation.robots
from rover_envs.utils.logging_utils import configure_datarecorder


def main():
    output_dir = args.output / os.environ.get("HOSTNAME", "local")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    cfg.scene.set_terrain(args.terrain)
    configure_datarecorder(cfg, str(output_dir), f"rover_seed_{args.seed}", "RL")
    env = gym.make(args.task, cfg=cfg)
    generator = torch.Generator(device=env.unwrapped.device).manual_seed(args.seed)
    try:
        env.reset(seed=args.seed)
        with torch.inference_mode():
            for step in range(args.steps):
                # Hold each bounded velocity/turn command for 10 steps.
                if step % 10 == 0:
                    action = torch.empty((1, 2), device=env.unwrapped.device)
                    if not args.varied:
                        action[:, 0] = 0.2 + 0.5 * torch.rand((1,), generator=generator, device=action.device)
                        action[:, 1] = torch.empty((1,), device=action.device).uniform_(-0.35, 0.35, generator=generator)
                    else:
                        mode = step // 10 % 4
                        speeds = ((0.2, 0.7), (-0.35, -0.1), (0.0, 0.0), (0.0, 0.0))
                        turns = ((-0.35, 0.35), (-0.25, 0.25), (-0.35, 0.35), (0.0, 0.0))
                        action[:, 0] = torch.empty((1,), device=action.device).uniform_(*speeds[mode], generator=generator)
                        action[:, 1] = torch.empty((1,), device=action.device).uniform_(*turns[mode], generator=generator)
                    if not torch.isfinite(action).all() or action[:, 0].abs().max() > 0.7 or action[:, 1].abs().max() > 0.35:
                        raise RuntimeError("Generated action is outside the recording bounds")
                env.step(action)
    finally:
        env.close()  # The RLRoverLab recorder exports the HDF5 file on close.
    output = output_dir / f"rover_seed_{args.seed}.hdf5"
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"RLRoverLab did not export a dataset at {output}")
    with h5py.File(output) as data:
        episodes = list(data["data"].values())
        recorded_actions = np.concatenate([episode["actions"][:] for episode in episodes])
        if len(recorded_actions) != args.steps or recorded_actions.shape[1:] != (2,):
            raise RuntimeError("Recorded action count or shape does not match the requested run")
        if args.varied and not (
            (recorded_actions[:, 0] > 0).any()
            and (recorded_actions[:, 0] < 0).any()
            and (recorded_actions[:, 0] == 0).any()
            and (recorded_actions[:, 1] > 0).any()
            and (recorded_actions[:, 1] < 0).any()
        ):
            raise RuntimeError("Recorded actions do not cover all motion modes")
    (output_dir / "metadata.json").write_text(json.dumps({
        "task": args.task, "seed": args.seed, "terrain": args.terrain,
        "steps": args.steps, "sweep_index": args.sweep_index,
        "action_policy": "varied" if args.varied else "forward",
        "episodes": len(episodes),
    }, indent=2) + "\n")
    print(f"Saved rover transitions to {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
