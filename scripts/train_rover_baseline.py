"""Train and evaluate goal-state dynamics from RLRoverLab HDF5 episodes."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import torch


def load_episodes(path):
    episodes = []
    with h5py.File(path) as data:
        for name in sorted(data["data"], key=lambda key: int(key.split("_")[-1])):
            group = data["data"][name]
            obs = group["obs"][:]
            actions = group["actions"][:]
            dones = group["dones"][:]
            if (obs.ndim != 2 or actions.ndim != 2 or actions.shape[1] != 2
                    or obs.shape[1] <= 5 or dones.shape not in ((len(obs),), (len(obs), 1))
                    or len(obs) != len(actions) or len(obs) < 3):
                raise ValueError(f"Invalid transition shapes in {name}")
            if (dones[:-1].any() or not np.isin(dones, [0, 1]).all()
                    or not np.isfinite(obs).all() or not np.isfinite(actions).all()):
                raise ValueError(f"Invalid values or a reset inside {name}")
            if not np.allclose(obs[1:, :2], actions[:-1], atol=1e-6, rtol=0):
                raise ValueError(f"Observation/action timing mismatch in {name}")
            # Reset goals are stale until the first step; next_obs also keeps stale goals.
            episodes.append((name, obs[1:-1].copy(), actions[1:-1].copy(), obs[2:].copy()))
    if not episodes:
        raise ValueError(f"No episodes in {path}")
    return episodes


def goal_delta(target, obs):
    delta = target[..., 2:5] - obs[..., 2:5]
    # Policy angles are radians / pi, so their period is two.
    return torch.cat((delta[..., :1], (delta[..., 1:] + 1).remainder(2) - 1), dim=-1)


def relative_motion(obs, target):
    # shortcut: assumes a fixed goal within each episode, record goal changes before longer runs.
    yaw = -goal_delta(target, obs)[:, 2] * torch.pi
    bearing = obs[:, 3] * torch.pi
    next_bearing = target[:, 3] * torch.pi + yaw
    return torch.stack((obs[:, 2] * bearing.cos() - target[:, 2] * next_bearing.cos(),
                        obs[:, 2] * bearing.sin() - target[:, 2] * next_bearing.sin(), yaw), dim=1)
class GoalDynamics(torch.nn.Module):
    def __init__(self, use_actions=True):
        super().__init__()
        self.use_actions = use_actions
        self.register_buffer("delta_scale", torch.ones(3))
        self.net = torch.nn.Sequential(torch.nn.Linear(4, 32), torch.nn.Tanh(), torch.nn.Linear(32, 3))

    def forward(self, inputs):
        obs, actions = inputs[:, :5], inputs[:, 5:7]
        commands = torch.cat((obs[:, :2], actions), dim=1)
        dx, dy, yaw = self.motion(commands).unbind(dim=1)
        bearing = obs[:, 3] * torch.pi
        x = obs[:, 2] * bearing.cos() - dx
        y = obs[:, 2] * bearing.sin() - dy
        distance = torch.sqrt(x.square() + y.square() + 1e-12)
        next_bearing = torch.atan2(y, x) / torch.pi - yaw / torch.pi
        heading = obs[:, 4] - yaw / torch.pi
        goals = torch.stack((distance, next_bearing, heading), dim=1)
        return torch.cat((actions - obs[:, :2], goals - obs[:, 2:5]), dim=1)

    def motion(self, commands):
        if not self.use_actions:
            commands = torch.zeros_like(commands)
        return self.net(commands) * self.delta_scale


def model_for(obs_dim, action_dim, use_actions=True):
    if obs_dim != 5 or action_dim != 2:
        raise ValueError("Goal dynamics needs five policy-state fields and two actions")
    return GoalDynamics(use_actions)


def predict(model, obs, actions):
    prediction = obs + model(torch.cat((obs, actions), dim=1))
    prediction[:, 2] = prediction[:, 2].clamp_min(0)
    prediction[:, 3:5] = (prediction[:, 3:5] + 1).remainder(2) - 1
    return prediction


def split_episodes(paths, validation_paths=None):
    paths = [path.resolve() for path in paths]
    validation_paths = [path.resolve() for path in (validation_paths or [])]
    if len(set(paths + validation_paths)) != len(paths + validation_paths):
        raise ValueError("Training and validation files must be distinct, with no duplicates")
    episodes = [(f"{path}:{name}", *arrays) for path in paths
                for name, *arrays in load_episodes(path)]
    if validation_paths:
        validation = [(f"{path}:{name}", *arrays) for path in validation_paths
                      for name, *arrays in load_episodes(path)]
    else:
        if len(episodes) < 2:
            raise ValueError("Need at least two episodes for an episode-level validation split")
        episodes, validation = episodes[:-1], episodes[-1:]
    shapes = {(episode[1].shape[1], episode[2].shape[1]) for episode in episodes + validation}
    if len(shapes) != 1:
        raise ValueError("All recordings must use the same observation and action dimensions")
    return episodes, validation


def tensors(episodes):
    # The 10,201 scan values overwhelmed the goal loss; this baseline learns goal dynamics only.
    return tuple(torch.from_numpy(np.concatenate([episode[index][:, :5] if index != 2 else episode[index]
                                                   for episode in episodes]).astype(np.float32))
                 for index in (1, 2, 3))


def train(model, obs, actions, next_obs, steps):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    # Fit motion in the rover frame, independent of the arbitrary goal direction.
    delta = relative_motion(obs, next_obs)
    model.delta_scale.copy_(delta.std(dim=0, unbiased=False).clamp_min(1e-4))
    model.train()
    for _ in range(steps):
        batch = torch.randint(len(obs), (min(32, len(obs)),))
        commands = torch.cat((obs[batch, :2], actions[batch]), dim=1)
        loss = ((model.motion(commands) - delta[batch]) / model.delta_scale).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()


def mse(model, obs, actions, next_obs):
    with torch.inference_mode():
        return goal_delta(predict(model, obs, actions), next_obs).square().mean().item()


def rollout_mse(model, episodes, horizon=5):
    error = copy_error = count = 0
    with torch.inference_mode():
        for episode in episodes:
            obs, actions, next_obs = tensors([episode])
            starts = len(obs) - horizon + 1
            if starts < 1:
                continue
            prediction = obs[:starts].clone()
            for step in range(horizon):
                prediction = predict(model, prediction, actions[step:step + starts])
            target = next_obs[horizon - 1:]
            error += goal_delta(prediction, target).square().sum().item()
            copy_error += goal_delta(obs[:starts], target).square().sum().item()
            count += starts * 3
    return {"horizon": horizon, "mse": error / count if count else None,
            "copy_mse": copy_error / count if count else None, "elements": count}


def evaluate(model, episodes):
    obs, actions, target = tensors(episodes)
    with torch.inference_mode():
        shuffled = actions[torch.randperm(len(actions), generator=torch.Generator().manual_seed(0))]
        errors = goal_delta(predict(model, obs, actions), target).square().mean(dim=0)
        copy_errors = goal_delta(obs, target).square().mean(dim=0)
        return {"episodes": [episode[0] for episode in episodes], "transitions": len(obs),
                "goal_mse": errors.mean().item(), "copy_goal_mse": copy_errors.mean().item(),
                "per_field_mse": errors.tolist(), "per_field_copy_mse": copy_errors.tolist(),
                "shuffled_action_goal_mse": mse(model, obs, shuffled, target),
                "rollout": rollout_mse(model, episodes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", default=[Path("data/rover/rover_seed_0.hdf5")])
    parser.add_argument("--validation-data", type=Path, nargs="+",
                        help="Hold out whole recording files; otherwise hold out the last episode")
    parser.add_argument("--test-data", type=Path, nargs="+", help="Separate files scored only after training")
    parser.add_argument("--output", type=Path, default=Path("runs/rover-baseline"))
    args = parser.parse_args()
    all_paths = [path.resolve() for path in args.data + (args.validation_data or []) + (args.test_data or [])]
    if len(set(all_paths)) != len(all_paths):
        parser.error("Training, validation, and test files must be distinct")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    train_episodes, validation = split_episodes(args.data, args.validation_data)
    obs, actions, next_obs = tensors(train_episodes)
    val_obs, val_actions, val_next_obs = tensors(validation)

    tiny = model_for(obs.shape[1], actions.shape[1])
    train(tiny, obs[:32], actions[:32], next_obs[:32], steps=1000)
    tiny_mse = mse(tiny, obs[:32], actions[:32], next_obs[:32])
    tiny_copy_mse = goal_delta(obs[:32], next_obs[:32]).square().mean().item()
    if tiny_mse >= tiny_copy_mse:
        raise RuntimeError("The model did not overfit 32 transitions; check the data and training path")

    torch.manual_seed(0)
    model = model_for(obs.shape[1], actions.shape[1])
    train(model, obs, actions, next_obs, steps=1000)
    torch.manual_seed(0)
    no_action = model_for(obs.shape[1], actions.shape[1], use_actions=False)
    train(no_action, obs, actions, next_obs, steps=1000)
    report = {
        "data": [str(path) for path in args.data],
        "train_episodes": [episode[0] for episode in train_episodes],
        "split": "recording" if args.validation_data else "episode",
        "train_transitions": len(obs),
        "validation_transitions": len(val_obs),
        "tiny_batch_mse": tiny_mse,
        "tiny_batch_copy_mse": tiny_copy_mse,
        "train_mse": mse(model, obs, actions, next_obs),
        "validation": evaluate(model, validation),
        "validation_no_action": evaluate(no_action, validation),
        "seed": 0,
        "tiny_steps": 1000,
        "train_steps": 1000,
        "schema": "goal-motion-v1",
        "motion_fields": ["forward_scaled_0.11", "lateral_scaled_0.11", "yaw_radians"],
        "fields": ["distance_scaled_0.11", "bearing_div_pi", "heading_error_div_pi"],
        "reset_transitions_omitted": len(train_episodes) + len(validation),
        "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "h5py": h5py.__version__},
    }
    if args.test_data:
        test_episodes = [(f"{path.resolve()}:{name}", *arrays) for path in args.test_data
                         for name, *arrays in load_episodes(path)]
        if any(episode[1].shape[1] != train_episodes[0][1].shape[1] for episode in test_episodes):
            raise ValueError("Test observation dimensions do not match training")
        report["test"] = evaluate(model, test_episodes)
        report["test_no_action"] = evaluate(no_action, test_episodes)
        report["reset_transitions_omitted"] += len(test_episodes)
    run_dir = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    report["output"] = str(run_dir)
    shutil.copyfile(__file__, run_dir / Path(__file__).name)
    torch.save({"model": model.state_dict(), "obs_dim": obs.shape[1], "action_dim": actions.shape[1],
                "schema": report["schema"], "use_actions": True}, run_dir / "model.pt")
    torch.save({"model": no_action.state_dict(), "obs_dim": 5, "action_dim": 2,
                "schema": report["schema"], "use_actions": False}, run_dir / "no-action.pt")
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
