"""Record the upstream scripted rover action, RGB video, and world positions."""
import argparse
from contextlib import nullcontext
import csv
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="AAURoverEnvSimple-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--output", type=Path, default=Path("/data/runs/rover-video"))
parser.add_argument("--no-video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps < 1 or args.num_envs != 1:
    parser.error("Recording requires positive steps and exactly one environment")
args.enable_cameras = not args.no_video
if args.no_video:
    args.kit_args = ""
app = AppLauncher(args).app

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from isaaclab_tasks.utils import parse_env_cfg
import rover_envs.envs.navigation.robots


def main():
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = 0
    cfg.video_recorder.window_width = 640
    cfg.video_recorder.window_height = 480
    env = gym.make(args.task, cfg=cfg, render_mode=None if args.no_video else "rgb_array")
    rows = []
    frames = 0
    try:
        env.reset(seed=0)
        robot = env.unwrapped.scene["robot"]
        dt = float(env.unwrapped.step_dt)
        # Initialize the recorder's camera, then track the actual rover spawn.
        if not args.no_video:
            from isaacsim.core.rendering_manager import ViewportManager
            env.render()
        video = nullcontext() if args.no_video else imageio.get_writer(str(args.output / "rover.mp4"), fps=round(1 / dt), codec="libx264")
        with video as writer:
            for step in range(args.steps):
                with torch.inference_mode():
                    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
                    action[..., 0] = 1.0
                    _, reward, terminated, truncated, _ = env.step(action)
                    position = robot.data.root_pos_w[0].detach().cpu().numpy().copy()
                    reset = bool((terminated | truncated)[0].item())
                    rows.append([step + 1, (step + 1) * dt, *position.tolist(), float(reward[0]), reset])
                if not args.no_video:
                    target = position + np.array([0.0, 0.0, 0.6])
                    ViewportManager.set_camera_view("/OmniverseKit_Persp", eye=(position + [-5.0, -5.0, 3.5]).tolist(), target=target.tolist())
                    frame = env.render()
                    if frame is not None and np.ptp(frame) > 0:
                        writer.append_data(frame)
                        if frames == 0:
                            imageio.imwrite(args.output / "preview.png", frame)
                        frames += 1
        if frames == 0 and not args.no_video:
            raise RuntimeError("Renderer returned no visible frames")
        with (args.output / "trajectory.csv").open("w") as handle:
            writer = csv.writer(handle)
            writer.writerow(["step", "time_s", "x_m", "y_m", "z_m", "reward", "reset"])
            writer.writerows(rows)
        summary = {"task": args.task, "steps": len(rows), "frames": frames,
                   "simulation_seconds": args.steps * dt, "seed": 0,
                   "controller": "Scripted forward command; no learned policy or world model",
                   "resets": sum(row[-1] for row in rows),
                   "capture_mode": "position_trace" if args.no_video else "rgb_video"}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
