"""Record one CPU-only MarsExplorer episode and an SVG path plot."""

import argparse
import json
from pathlib import Path
import random

import gymnasium as gym
import mars_explorer  # Registers explorer-v1.
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("runs/explorer"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    # Upstream map generation may use global RNGs in addition to Gym's RNG.
    random.seed(args.seed)
    np.random.seed(args.seed)
    env = gym.make("explorer-v1")
    try:
        env.action_space.seed(args.seed)
        obs, _ = env.reset(seed=args.seed)
        observations = [obs.copy()]
        positions = [(env.unwrapped.x, env.unwrapped.y)]
        actions, rewards, terminated_flags, truncated_flags = [], [], [], []
        # Upstream truncates after maxSteps, rather than at maxSteps.
        limit = int(env.unwrapped.maxSteps) + 2
        for _ in range(limit):
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, _ = env.step(action)
            observations.append(obs.copy())
            positions.append((env.unwrapped.x, env.unwrapped.y))
            actions.append(action)
            rewards.append(reward)
            terminated_flags.append(terminated)
            truncated_flags.append(truncated)
            if terminated or truncated:
                break
        else:
            raise RuntimeError("Upstream episode did not end within its configured step limit")
        np.savez_compressed(
            args.output / "trajectory.npz",
            observations=np.stack(observations),
            privileged_grid_positions=np.asarray(positions),
            actions=np.asarray(actions), rewards=np.asarray(rewards),
            terminated=np.asarray(terminated_flags), truncated=np.asarray(truncated_flags),
            step_indices=np.arange(len(actions)),
        )
        metadata = {
            "environment": "explorer-v1", "seed": args.seed,
            "transitions": len(actions), "observation_alignment": "observations[t], actions[t], observations[t+1]",
            "actions": {"0": "+x", "1": "-x", "2": "+y", "3": "-y"},
            "coordinates": "Simulator grid indices, privileged diagnostic state",
            "time": "Discrete steps; physical seconds per step are unspecified",
            "task": "Exploration coverage, not point-goal navigation",
            "collision": bool(env.unwrapped.collision),
            "out_of_bounds": bool(env.unwrapped.out_of_bounds),
            "terminated": bool(terminated), "truncated": bool(truncated),
            "limitations": "No wheel slip, traction, slope, or 3D rover dynamics",
        }
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        points = " ".join(f"{x * 8 + 4},{y * 8 + 4}" for x, y in positions)
        width, height = env.unwrapped.sizeX * 8, env.unwrapped.sizeY * 8
        (args.output / "trajectory.svg").write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="white"/>'
            f'<polyline points="{points}" fill="none" stroke="black" stroke-width="2"/>'
            '</svg>\n'
        )
        print(json.dumps(metadata, indent=2))
        print(f"Saved trajectory and path plot to {args.output}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
