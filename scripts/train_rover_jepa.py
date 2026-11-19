"""Train Meta's DINO-WM predictor on frozen, episode-aligned rover recordings."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from scripts.train_rover_baseline import load_episodes, model_for, predict


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_images(scan):
    if scan.ndim != 2 or scan.shape[1] != 10201 or not torch.isfinite(scan).all():
        raise ValueError("Expected finite 101 by 101 rover scans")
    # Fixed physical scale preserves slope and height changes between observations.
    height = (-scan.reshape(-1, 1, 101, 101)).clamp(-2, 2) / 4 + 0.5
    image = F.interpolate(height, (112, 112), mode="bilinear", align_corners=False).expand(-1, 3, -1, -1)
    return (image - image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]) / image.new_tensor(
        [0.229, 0.224, 0.225])[None, :, None, None]


def encode_state(observation):
    bearing, heading = observation[..., 3] * torch.pi, observation[..., 4] * torch.pi
    return torch.cat((observation[..., :2],
                      torch.stack((observation[..., 2] * bearing.cos(), observation[..., 2] * bearing.sin(),
                                   heading.cos(), heading.sin()), dim=-1)), dim=-1)


def decode_goal(state):
    x, y, cosine, sine = state[..., 2:].unbind(dim=-1)
    return torch.stack(((x.square() + y.square() + 1e-12).sqrt(),
                        torch.atan2(y, x) / torch.pi, torch.atan2(sine, cosine) / torch.pi), dim=-1)


def goal_error(predicted, target):
    delta = decode_goal(predicted) - decode_goal(target)
    return torch.cat((delta[..., :1], (delta[..., 1:] + 1).remainder(2) - 1), dim=-1).square().mean()


def prior_motion(prior, previous, actions):
    motion = actions.new_zeros((len(actions), 3))
    if prior is None:
        return motion
    for command in actions.reshape(-1, 5, 2).unbind(dim=1):
        local = prior.motion(torch.cat((previous, command), dim=-1))
        cosine, sine = motion[:, 2].cos(), motion[:, 2].sin()
        motion = torch.stack((motion[:, 0] + cosine * local[:, 0] - sine * local[:, 1],
                              motion[:, 1] + sine * local[:, 0] + cosine * local[:, 1],
                              motion[:, 2] + local[:, 2]), dim=-1)
        previous = command
    return motion


def advance_goal(state, motion):
    xy = state[:, 2:4] - motion[:, :2]
    heading = state[:, 4:6]
    cosine, sine = motion[:, 2].cos(), motion[:, 2].sin()
    return torch.stack((cosine * xy[:, 0] + sine * xy[:, 1],
                        -sine * xy[:, 0] + cosine * xy[:, 1],
                        cosine * heading[:, 0] + sine * heading[:, 1],
                        -sine * heading[:, 0] + cosine * heading[:, 1]), dim=-1)


def warp_features(features, motion):
    cosine, sine = motion[:, 2].cos(), motion[:, 2].sin()
    # align_corners=False treats 101 samples at 5 cm spacing as a 5.05 m raster.
    extent = 101 * 0.05
    theta = torch.stack((cosine, -sine, 2 * motion[:, 0] / (extent * 0.11),
                         sine, cosine, 2 * motion[:, 1] / (extent * 0.11)), dim=-1).reshape(-1, 2, 3)
    patches = features.transpose(1, 2).reshape(len(features), -1, 8, 8)
    grid = F.affine_grid(theta, patches.shape, align_corners=False)
    # shortcut: static-terrain transport approximates DINO patches, validate new scene types before use.
    return F.grid_sample(patches, grid, padding_mode="border", align_corners=False).flatten(2).transpose(1, 2)


def sequence_arrays(path, stride=5):
    result = []
    for name, obs, actions, next_obs in load_episodes(path):
        states = np.concatenate((obs[:1], next_obs))
        indices = np.arange(0, len(actions) + 1, stride)
        if len(indices) < 7:
            continue
        if states.shape[1] != 10206:
            raise ValueError(f"Unexpected scan schema in {path}:{name}")
        blocks = np.stack([actions[i:i + stride].reshape(-1) for i in indices[:-1]])
        if not np.isfinite(states).all() or not np.isfinite(blocks).all():
            raise ValueError("Non-finite sequence")
        result.append((f"{path}:{name}", states[indices], blocks))
    if not result:
        raise ValueError(f"No sufficiently long sequences in {path}")
    return result


def encode_episodes(paths, encoder, device):
    episodes = []
    for path in paths:
        for name, states, actions in sequence_arrays(path):
            encoded = []
            with torch.inference_mode():
                for start in range(0, len(states), 16):
                    scans = torch.from_numpy(states[start:start + 16, 5:].astype(np.float32)).to(device)
                    encoded.append(encoder.forward_features(scan_images(scans))["x_norm_patchtokens"].cpu())
            features = torch.cat(encoded)
            if features.shape[1:] != (64, 384) or not torch.isfinite(features).all():
                raise RuntimeError("Invalid DINO features")
            episodes.append((name, features, encode_state(torch.tensor(states[:, :5], dtype=torch.float32)),
                             torch.tensor(actions, dtype=torch.float32)))
    return episodes


def windows(episodes, mean, scale, projection=None, latent_scale=None):
    result = [[], [], [], []]
    for _, features, state, actions in episodes:
        features = (features - mean) / scale
        latent = features if projection is None else (features @ projection) / latent_scale
        for start in range(len(features) - 6):
            result[0].append(latent[start:start + 7])
            result[1].append(state[start:start + 7])
            result[2].append(actions[start:start + 6])
            result[3].append(features[start:start + 7])
    if not result[0]:
        raise ValueError("No complete two-context, five-future windows")
    return tuple(torch.stack(values) for values in result)


class RoverJEPA(nn.Module):
    def __init__(self, predictor_path, depth=2, motion_prior=None):
        super().__init__()
        spec = importlib.util.spec_from_file_location("meta_rover_vit", predictor_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.action_encoder = nn.Linear(10, 16)
        self.state_encoder = nn.Linear(2, 16)
        self.motion_prior = motion_prior
        self.predictor = module.ViTPredictor(num_patches=64, num_frames=2, dim=96,
                                           depth=depth, heads=4, dim_head=24, mlp_dim=192,
                                           dropout=0.1, use_sdpa=True)
        self.readout = nn.Linear(96, 64)
        # Start near persistence, a strong baseline for adjacent height scans.
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)

    def forward(self, features, states, actions):
        motion = prior_motion(self.motion_prior, states[:, -1, :2], actions[:, -1])
        past_motion = prior_motion(self.motion_prior, states[:, 0, :2], actions[:, 0])
        transported = warp_features(features[:, -1], motion)
        aligned_past = warp_features(warp_features(features[:, 0], past_motion), motion)
        aligned = torch.stack((aligned_past, transported), dim=1)
        act = self.action_encoder(actions / actions.new_tensor([0.7, 0.35] * 5))
        state = self.state_encoder(states[..., :2] / states.new_tensor([0.7, 0.35]))
        inputs = torch.cat((aligned, state[:, :, None].expand(-1, -1, 64, -1),
                            act[:, :, None].expand(-1, -1, 64, -1)), dim=-1)
        predicted = self.predictor(inputs.flatten(1, 2)).reshape(-1, 2, 64, 96)
        return (transported + self.readout(predicted[:, -1]),
                advance_goal(states[:, -1], motion))


def rollout(model, features, states, actions, steps):
    prediction = features[:, :2]
    context_states = states[:, :2]
    outputs, output_states = [], []
    for step in range(steps):
        next_features, next_pose = model(prediction[:, -2:], context_states, actions[:, step:step + 2])
        # Roll out predicted state; never use future recorded state as an input.
        next_state = torch.cat((actions[:, step + 1, -2:], next_pose), dim=-1)
        outputs.append(next_features)
        output_states.append(next_state)
        prediction = torch.cat((prediction[:, -1:], next_features[:, None]), dim=1)
        context_states = torch.cat((context_states[:, -1:], next_state[:, None]), dim=1)
    return torch.stack(outputs, dim=1), torch.stack(output_states, dim=1)


@torch.inference_mode()
def evaluate(model, dataset, device, projection, latent_scale, baseline=None):
    model.eval()
    scores = {"one_step_mse": 0., "five_step_mse": 0., "copy_one_step_mse": 0.,
              "copy_five_step_mse": 0., "shuffled_one_step_mse": 0., "shuffled_five_step_mse": 0.}
    for name in ("one", "five"):
        for prefix in ("goal_", "goal_copy_", "goal_shuffled_"):
            scores[f"{prefix}{name}_step_mse"] = 0.
        if baseline is not None:
            scores[f"goal_baseline_{name}_step_mse"] = 0.
            scores[f"transport_{name}_step_mse"] = 0.
    count = 0
    projection, latent_scale = projection.to(device), latent_scale.to(device)
    # Shuffle across the whole split, not adjacent windows from the same episode.
    permutation = torch.randperm(len(dataset[0]), generator=torch.Generator().manual_seed(0))
    wrong_actions = dataset[2].clone()
    wrong_actions[:, 1:] = dataset[2][permutation, 1:]
    for start in range(0, len(dataset[0]), 32):
        features, states, actions, full_features = [value[start:start + 32].to(device) for value in dataset]
        batch = len(features)
        predictions, poses = rollout(model, features, states, actions, 5)
        shuffled_predictions, shuffled_poses = rollout(model, features, states,
                                                      wrong_actions[start:start + 32].to(device), 5)
        baseline_states = {}
        transported = {}
        if baseline is not None:
            transport_features = features[:, 1]
            previous_command = states[:, 1, :2]
            for step in range(5):
                movement = prior_motion(baseline, previous_command, actions[:, step + 1])
                transport_features = warp_features(transport_features, movement)
                previous_command = actions[:, step + 1, -2:]
                transported[step + 1] = transport_features
            baseline_prediction = torch.cat((states[:, 1, :2], decode_goal(states[:, 1])), dim=-1)
            for control_step in range(25):
                block, offset = 1 + control_step // 5, control_step % 5
                baseline_prediction = predict(baseline, baseline_prediction, actions[:, block, 2 * offset:2 * offset + 2])
                if control_step in (4, 24):
                    baseline_states[(control_step + 1) // 5] = encode_state(baseline_prediction)
        for steps, name in ((1, "one"), (5, "five")):
            target = full_features[:, steps + 1]
            # Score all 384 original features; persist the unmodelled components.
            pred = full_features[:, 1] + ((predictions[:, steps - 1] - features[:, 1]) * latent_scale) @ projection.T
            shuffled = full_features[:, 1] + ((shuffled_predictions[:, steps - 1] - features[:, 1]) * latent_scale) @ projection.T
            scores[f"{name}_step_mse"] += F.mse_loss(pred, target).item() * batch
            scores[f"copy_{name}_step_mse"] += F.mse_loss(full_features[:, 1], target).item() * batch
            scores[f"shuffled_{name}_step_mse"] += F.mse_loss(shuffled, target).item() * batch
            pose_target = states[:, steps + 1]
            scores[f"goal_{name}_step_mse"] += goal_error(poses[:, steps - 1], pose_target).item() * batch
            scores[f"goal_copy_{name}_step_mse"] += goal_error(states[:, 1], pose_target).item() * batch
            scores[f"goal_shuffled_{name}_step_mse"] += goal_error(shuffled_poses[:, steps - 1], pose_target).item() * batch
            if baseline is not None:
                scores[f"goal_baseline_{name}_step_mse"] += goal_error(baseline_states[steps], pose_target).item() * batch
                transport_full = full_features[:, 1] + ((transported[steps] - features[:, 1]) * latent_scale) @ projection.T
                scores[f"transport_{name}_step_mse"] += F.mse_loss(transport_full, target).item() * batch
        count += batch
    scores = {key: value / count for key, value in scores.items()}
    if not all(math.isfinite(value) for value in scores.values()):
        raise RuntimeError("Non-finite evaluation")
    scores["windows"] = count
    scores["mean_copy_ratio"] = sum(scores[f"{name}_step_mse"] / max(scores[f"copy_{name}_step_mse"], 1e-12)
                                    for name in ("one", "five")) / 2
    scores["mean_goal_copy_ratio"] = sum(scores[f"goal_{name}_step_mse"] / max(scores[f"goal_copy_{name}_step_mse"], 1e-12)
                                         for name in ("one", "five")) / 2
    scores["selection_score"] = scores["mean_copy_ratio"]
    return scores


def quality_passed(scores):
    scan_passed = all(scores[f"{name}_step_mse"] < 0.9 * scores[f"copy_{name}_step_mse"]
               and scores[f"shuffled_{name}_step_mse"] > 1.02 * scores[f"{name}_step_mse"]
               and scores[f"{name}_step_mse"] < 0.98 * scores[f"transport_{name}_step_mse"]
               for name in ("one", "five"))
    pose_passed = all(scores[f"goal_{name}_step_mse"] < 0.9 * scores[f"goal_copy_{name}_step_mse"]
                      and scores[f"goal_shuffled_{name}_step_mse"] > 1.02 * scores[f"goal_{name}_step_mse"]
                      for name in ("one", "five"))
    return scan_passed and pose_passed


def update_plateau(score, reference, stale):
    if score < reference * 0.999:
        return score, 0
    return reference, stale + 1


def forecast_loss(prediction, target, latent_scale, reference):
    error = ((prediction - target) * latent_scale).square().mean(dim=(0, 2, 3))
    return (error / reference).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-data", type=Path, nargs="+", required=True)
    parser.add_argument("--test-data", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictor-source", type=Path, required=True)
    parser.add_argument("--encoder-source", type=Path, required=True)
    parser.add_argument("--motion-baseline", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--rollout-steps", type=int, default=1, choices=(1, 2, 5))
    args = parser.parse_args()
    paths = args.data + args.validation_data + args.test_data
    if len({path.resolve() for path in paths}) != len(paths) or args.epochs < 1 or args.patience < 1:
        parser.error("Use distinct recording files and positive epoch/patience counts")
    if not torch.cuda.is_available():
        raise RuntimeError("The training job requires CUDA")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.set_num_threads(2)
    device = torch.device("cuda")
    encoder = torch.hub.load(str(args.encoder_source), "dinov2_vits14", source="local", pretrained=True)
    encoder = encoder.to(device).eval().requires_grad_(False)
    # Test recordings stay unopened until checkpoint selection is complete.
    train_episodes = encode_episodes(args.data, encoder, device)
    val_episodes = encode_episodes(args.validation_data, encoder, device)
    features = torch.cat([episode[1] for episode in train_episodes])
    mean = features.mean(dim=(0, 1))
    scale = features.std(dim=(0, 1), unbiased=False).clamp_min(0.1)
    feature_std = features.std(dim=0, unbiased=False).mean().item()
    if feature_std < 0.01:
        raise RuntimeError("Encoder features have insufficient variation")
    normalized = ((features - mean) / scale).flatten(0, 1)
    _, _, projection = torch.pca_lowrank(normalized, q=64, center=False, niter=4)
    latent_scale = (normalized @ projection).std(dim=0, unbiased=False).clamp_min(0.1)
    retained_variance = (normalized @ projection).square().sum().item() / normalized.square().sum().item()
    train_data, val_data = [windows(episodes, mean, scale, projection, latent_scale)
                           for episodes in (train_episodes, val_episodes)]
    copy_reference = ((train_data[0][:, 2:] - train_data[0][:, 1:2]) * latent_scale).square().mean(
        dim=(0, 2, 3)).clamp_min(1e-6).to(device)
    baseline = model_for(5, 2).to(device).eval().requires_grad_(False)
    baseline.load_state_dict(torch.load(args.motion_baseline, map_location=device, weights_only=True)["model"])
    model = RoverJEPA(args.predictor_source, motion_prior=baseline).to(device)
    state_scale = torch.cat([ep[2][1:, 2:] - ep[2][:-1, 2:] for ep in train_episodes]).std(
        dim=0, unbiased=False).clamp_min(0.001).to(device)
    manifest = {"model": "rover-dino-wm-v6", "encoder": "dinov2_vits14 pretrained, frozen; train-only PCA to 64 features",
                "predictor": "Meta JEPA-WMs ViTPredictor; depth 2, width 96, 4 heads, residual readout",
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "motion_prior": "Frozen GoalDynamics trained on this run split; learns forward, lateral, and yaw motion",
                "transport": "Frozen motion aligns both observations to the predicted next frame; JEPA learns feature corrections",
                "transport_raster_extent_m": 5.05,
                "training_objective": "Per-horizon feature error relative to training-copy error; motion prior frozen",
                "training_copy_error_by_horizon": copy_reference.cpu().tolist(),
                "input": "112x112 height grid; previous command; goal-relative pose; five applied commands",
                "motion_baseline_sha256": file_hash(args.motion_baseline),
                "state_targets": "goal vector x/y (distance scaled 0.11); heading-error cosine/sine",
                "context_frames": 2, "frame_stride": 5, "seconds_per_frame": 1.,
                "train_windows": len(train_data[0]), "validation_windows": len(val_data[0]),
                "feature_std": feature_std, "seed": 0, "learning_rate": args.learning_rate,
                "retained_training_feature_variance": retained_variance,
                "evaluation": "All 384 original features; persist components outside the training PCA subspace",
                "training_rollout_steps": args.rollout_steps, "max_epochs": args.epochs, "patience": args.patience,
                "files": {str(path): file_hash(path) for path in paths},
                "source_hashes": {str(path): file_hash(path) for path in (Path(__file__), args.predictor_source)},
                "torch": str(torch.__version__), "numpy": np.__version__}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    best, plateau_reference, stale, stopped = math.inf, math.inf, 0, "epoch_limit"
    history = []
    for epoch in range(args.epochs):
        start_time = time.monotonic()
        model.train()
        total = count = scan_total = pose_total = 0
        for batch in torch.randperm(len(train_data[0])).split(16):
            obs, states, actions = [value[batch].to(device) for value in train_data[:3]]
            prediction, pose = rollout(model, obs, states, actions, args.rollout_steps)
            scan_loss = forecast_loss(prediction, obs[:, 2:2 + args.rollout_steps], latent_scale.to(device),
                                      copy_reference[:args.rollout_steps])
            pose_loss = ((pose[..., 2:] - states[:, 2:2 + args.rollout_steps, 2:]) / state_scale).square().mean()
            loss = scan_loss
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(batch)
            scan_total += scan_loss.item() * len(batch)
            pose_total += pose_loss.item() * len(batch)
            count += len(batch)
        scores = evaluate(model, val_data, device, projection, latent_scale, baseline)
        row = {"epoch": epoch + 1, "train_loss": total / count,
               "train_scan_loss": scan_total / count, "frozen_motion_loss": pose_total / count,
               "seconds": time.monotonic() - start_time, "validation": scores}
        history.append(row)
        print(json.dumps(row), flush=True)
        with (args.output / "history.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        if scores["selection_score"] < best or quality_passed(scores):
            best = scores["selection_score"]
            torch.save({"model": model.state_dict(), "mean": mean, "scale": scale,
                        "projection": projection, "latent_scale": latent_scale, "state_scale": state_scale.cpu(),
                        "epoch": epoch + 1, "validation": scores, "manifest": manifest}, args.output / "best.pt")
        plateau_reference, stale = update_plateau(scores["selection_score"], plateau_reference, stale)
        if epoch >= 4 and quality_passed(scores):
            stopped = "validation_target_met"
            break
        if stale >= args.patience:
            stopped = "validation_plateau"
            break
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    # Re-evaluation checks that the saved model reproduces its selected metrics.
    validation = evaluate(model, val_data, device, projection, latent_scale, baseline)
    if abs(validation["selection_score"] - checkpoint["validation"]["selection_score"]) > 1e-5:
        raise RuntimeError("Checkpoint reload changed validation results")
    test_episodes = encode_episodes(args.test_data, encoder, device)
    test = evaluate(model, windows(test_episodes, mean, scale, projection, latent_scale), device,
                    projection, latent_scale, baseline)
    report = {"stop_reason": stopped, "epochs": len(history), "best_epoch": checkpoint["epoch"],
              "validation": validation, "test": test,
              "quality_passed": quality_passed(validation) and quality_passed(test),
              "scope": "Offline future-feature prediction; not closed-loop navigation or safety evidence"}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "FINISHED").touch()
    if report["quality_passed"]:
        (args.output / "QUALITY_PASSED").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
