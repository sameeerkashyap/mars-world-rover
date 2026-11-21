"""Compare frozen-encoder AdaLN and DINO-WM on the same recorded rover windows."""

import argparse
import json
import math
from pathlib import Path
import random

import numpy as np
import torch

from scripts.train_rover_baseline import model_for
from scripts.train_rover_jepa import RoverJEPA, encode_episodes, file_hash, prior_motion, rollout, update_plateau, warp_features
from src.rover_training import WindowDataset, atomic_save, cached_episodes, feature_statistics, restore_progress, save_progress
from src.rover_world_model import RoverWorldModel


def selected_indices(dataset, limit):
    return torch.linspace(0, len(dataset) - 1, min(limit, len(dataset))).long()


def full_batch(dataset, indices, device):
    _, states, actions, features = dataset.batch(indices)
    return features.to(device), states[:, :2, :2].to(device), actions.to(device)


def action_donors(dataset, indices):
    episodes = [dataset.location(index)[0][0] for index in indices]
    generator = torch.Generator().manual_seed(0)
    donors = []
    for episode in episodes:
        eligible = [index for index, name in enumerate(episodes) if name != episode]
        if not eligible:
            raise ValueError("Action-use evaluation needs sampled windows from at least two episodes")
        donors.append(eligible[int(torch.randint(len(eligible), (1,), generator=generator))])
    return indices[donors]


def candidate_passed(scores):
    return (all(value >= 0.02 for value in scores["gain_over_transport"])
            and all(value > 0 for value in scores["action_shuffle_increase"])
            and all(model < control for key in ("copy", "current_residual")
                    for model, control in zip(scores["model"], scores[key])))


def load_residual(path, upstream, device):
    if file_hash(path) != "e28f932c1c5802556aa4432eba63c0c2c5aa82799ddde2f7bcaa42945d2a9079":
        raise ValueError("The frozen current-residual checkpoint changed")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["manifest"]["model"] != "rover-dino-wm-v6":
        raise ValueError("Expected the current transported-residual model")
    model = RoverJEPA(upstream / "app/plan_common/models/vit.py", motion_prior=model_for(5, 2)).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval().requires_grad_(False)
    return model, {key: checkpoint[key].to(device) for key in ("mean", "scale", "projection", "latent_scale")}


def residual_forecasts(residual, features, previous, blocks, normalization):
    model, norm = residual
    scale, mean = normalization["scale"], normalization["mean"]
    raw = features[:, :2] * scale + mean
    context = ((raw - norm["mean"]) / norm["scale"]) @ norm["projection"] / norm["latent_scale"]
    states = features.new_zeros((len(features), 2, 6))
    states[..., :2], states[..., 4] = previous, 1.
    prediction = rollout(model, context, states, blocks, 5)[0]
    transported, last_command, control = context[:, 1], previous[:, -1], []
    for step in range(5):
        transported = warp_features(transported, prior_motion(model.motion_prior, last_command, blocks[:, step + 1]))
        last_command = blocks[:, step + 1, -2:]
        control.append(transported)
    forecasts = {"current_residual": prediction, "current_transport": torch.stack(control, dim=1)}
    return {key: features[:, 1:2] + (((value - context[:, 1:2]) * norm["latent_scale"]) @ norm["projection"].T)
            * norm["scale"] / scale for key, value in forecasts.items()}


@torch.inference_mode()
def evaluate(model, dataset, indices, scale, prior, device, residual=None, normalization=None, recordings=None):
    model.eval()
    errors = {key: [] for key in ("model", "copy", "transport", "shuffled")}
    objective_errors = []
    if residual is not None:
        errors.update({"current_residual": [], "current_transport": []})
    donors = action_donors(dataset, indices)
    wrong = dataset.batch(donors)[2]
    for start in range(0, len(indices), 16):
        features, previous, blocks = full_batch(dataset, indices[start:start + 16], device)
        shuffled = blocks.clone()
        shuffled[:, 1:] = wrong[start:start + 16, 1:].to(device)
        prediction = model.forecast(features[:, :2], previous, blocks)
        objective_errors.append((prediction[:, :2] - features[:, 2:4]).square().mean(dim=(1, 2, 3)).cpu())
        perturbed = model.forecast(features[:, :2], previous, shuffled)
        transported = features[:, 1]
        last_command = previous[:, -1]
        forecasts = []
        for step in range(5):
            transported = warp_features(transported, prior_motion(prior, last_command, blocks[:, step + 1]))
            last_command = blocks[:, step + 1, -2:]
            forecasts.append(transported)
        target = features[:, [2, 6]]
        predictions = {"model": prediction[:, [0, 4]], "shuffled": perturbed[:, [0, 4]],
                       "transport": torch.stack(forecasts, dim=1)[:, [0, 4]], "copy": features[:, 1:2]}
        if residual is not None:
            predictions.update({key: value[:, [0, 4]] for key, value in
                                residual_forecasts(residual, features, previous, blocks, normalization).items()})
        for key, value in predictions.items():
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite {key} validation forecast")
            errors[key].append(((value - target) * scale).square().mean(dim=-1).cpu())
    errors = {key: torch.cat(value) for key, value in errors.items()}
    scores = {key: value.mean(dim=(0, 2)).tolist() for key, value in errors.items()}
    scores["normalized_training_objective"] = torch.cat(objective_errors).mean().item()
    center = torch.tensor([27, 28, 35, 36])
    scores["near_rover"] = {key: value[:, :, center].mean(dim=(0, 2)).tolist() for key, value in errors.items()}
    episodes = [dataset.location(index)[0][0] for index in indices]
    layouts = [(recordings or {}).get(name.rsplit(":", 1)[0], {}).get("terrain", name.rsplit(":", 1)[0]) for name in episodes]
    for field, names in (("by_episode", episodes), ("by_layout", layouts)):
        scores[field] = {name: {key: value[[n == name for n in names]].mean(dim=(0, 2)).tolist()
                               for key, value in errors.items()} for name in sorted(set(names))}
    scores["selection_score"] = sum(a / max(b, 1e-12) for a, b in zip(scores["model"], scores["transport"])) / 2
    scores["gain_over_transport"] = [1 - a / max(b, 1e-12) for a, b in zip(scores["model"], scores["transport"])]
    scores["action_shuffle_increase"] = [a / max(b, 1e-12) - 1 for a, b in zip(scores["shuffled"], scores["model"])]
    return scores


def train_arm(args, architecture, seed, train, validation, indices, val_indices, normalization, prior, manifest):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    output = args.output / f"{architecture}-seed-{seed}"
    output.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device(args.device)
    model = RoverWorldModel(architecture, args.upstream).to(device)
    identity = {**manifest, "architecture": architecture, "seed": seed,
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history, best, plateau, stale, first, stopped = [], math.inf, math.inf, 0, 0, "epoch_limit"
    if args.resume and (output / "last.pt").exists():
        saved = restore_progress(output / "last.pt", model, optimizer, identity)
        history, best, plateau, stale = saved["history"], saved["best"], saved["plateau"], saved["stale"]
        first, stopped = saved["epoch"], saved["stopped"]
    else:
        (output / "manifest.json").write_text(json.dumps(identity, indent=2) + "\n")
        tiny = indices[:64]
        features, previous, blocks = full_batch(train, tiny, device)
        model.eval()
        with torch.no_grad():
            initial = (model.forecast(features[:, :2], previous, blocks[:, :3], 2) - features[:, 2:4]).square().mean().item()
        model.train()
        for step in range(100):
            batch = torch.arange(len(tiny), device=device).roll(step * args.batch_size)[:args.batch_size]
            prediction = model.forecast(features[batch, :2], previous[batch], blocks[batch, :3], 2, detach=True)
            loss = (prediction - features[batch, 2:4]).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            final = (model.forecast(features[:, :2], previous, blocks[:, :3], 2) - features[:, 2:4]).square().mean().item()
        fit = {"windows": len(tiny), "updates": 100, "initial_loss": initial, "final_loss": final,
               "passed": math.isfinite(final) and final < initial * 0.5}
        (output / "fit-check.json").write_text(json.dumps(fit, indent=2) + "\n")
        if not fit["passed"]:
            raise RuntimeError(f"Tiny-window learning check failed: {architecture}, seed {seed}: {fit}")
        torch.manual_seed(seed)
        model = RoverWorldModel(architecture, args.upstream).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    for epoch in range(first, args.epochs):
        if stopped != "epoch_limit":
            break
        model.train()
        total = 0.
        for batch in indices[torch.randperm(len(indices))].split(args.batch_size):
            features, previous, blocks = full_batch(train, batch, device)
            prediction = model.forecast(features[:, :2], previous, blocks[:, :3], 2, detach=True)
            loss = (prediction - features[:, 2:4]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite pilot loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * len(batch)
        scores = evaluate(model, validation, val_indices, normalization["scale"].to(device), prior, device,
                          args.residual, {key: value.to(device) for key, value in normalization.items()}, manifest.get("recordings"))
        row = {"epoch": epoch + 1, "training_loss": total / len(indices), "validation": scores}
        history.append(row)
        print(json.dumps({"arm": output.name, **row}), flush=True)
        if scores["selection_score"] < best:
            best = scores["selection_score"]
            atomic_save({"model": model.state_dict(), "normalization": normalization,
                         "manifest": identity, "epoch": epoch + 1, "validation": scores}, output / "best.pt")
        plateau, stale = update_plateau(scores["selection_score"], plateau, stale)
        if stale >= args.patience:
            stopped = "validation_plateau"
        save_progress(output / "last.pt", model, optimizer, identity, epoch + 1, history, best, plateau, stale, stopped)
    saved = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    scores = evaluate(model, validation, val_indices, normalization["scale"].to(device), prior, device,
                      args.residual, {key: value.to(device) for key, value in normalization.items()}, manifest.get("recordings"))
    if abs(scores["selection_score"] - saved["validation"]["selection_score"]) > 1e-6:
        raise RuntimeError("Checkpoint reload changed the validation score")
    report = {"architecture": architecture, "seed": seed, "best_epoch": saved["epoch"],
              "epochs": len(history), "stop_reason": stopped, "validation": scores,
              "checkpoint_sha256": file_hash(output / "best.pt"),
              "candidate_gate": candidate_passed(scores)}
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-data", type=Path, nargs="+", required=True)
    parser.add_argument("--encoder-source", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--motion-baseline", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--train-windows", type=int, default=256)
    parser.add_argument("--validation-windows", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (min(args.epochs, args.patience, args.batch_size, args.train_windows, args.validation_windows) < 1
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds)):
        parser.error("Need positive training settings and distinct nonnegative seeds")
    paths = args.data + args.validation_data
    hashes = {str(path): file_hash(path) for path in paths}
    if len(hashes) != len(paths) or len(set(hashes.values())) != len(paths):
        parser.error("Training and validation recordings must have distinct paths and contents")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(2)
    device = torch.device(args.device)
    encoder = torch.hub.load(str(args.encoder_source), "dinov2_vits14", source="local", pretrained=True).to(device).eval().requires_grad_(False)
    encoder_hash = file_hash(Path(torch.hub.get_dir()) / "checkpoints/dinov2_vits14_pretrain.pth")
    if encoder_hash != "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9":
        raise ValueError("Frozen encoder weights changed")
    signature = {"weights": encoder_hash, "preprocessing": file_hash(Path(encode_episodes.__code__.co_filename)),
                 "loader": file_hash(Path(model_for.__code__.co_filename)),
                 "encoder_source": file_hash(args.encoder_source / "hubconf.py")}
    train_episodes = cached_episodes(args.data, encoder, device, args.cache_dir, signature, encode_episodes, file_hash)
    val_episodes = cached_episodes(args.validation_data, encoder, device, args.cache_dir, signature, encode_episodes, file_hash)
    del encoder
    mean, scale, projection, latent_scale, _, _ = feature_statistics(train_episodes, project=False)
    train, validation = [WindowDataset(episodes, mean, scale, projection, latent_scale) for episodes in (train_episodes, val_episodes)]
    indices, val_indices = selected_indices(train, args.train_windows), selected_indices(validation, args.validation_windows)
    if min(len(indices), len(val_indices)) < 2:
        raise ValueError("Pilot needs at least two training and validation windows")
    prior = model_for(5, 2).to(device).eval().requires_grad_(False)
    prior.load_state_dict(torch.load(args.motion_baseline, map_location=device, weights_only=True)["model"])
    args.residual = load_residual(args.residual_checkpoint, args.upstream, device)
    sources = [Path(__file__), Path(RoverWorldModel.forward.__code__.co_filename),
               Path(feature_statistics.__code__.co_filename), Path(encode_episodes.__code__.co_filename),
               Path(model_for.__code__.co_filename), *sorted(args.upstream.rglob("*.py"))]
    manifest = {"schema": "rover-world-model-pilot-v2", "files": hashes, "encoder_weights": encoder_hash,
                "training_files": [str(path) for path in args.data], "validation_files": [str(path) for path in args.validation_data],
                "current_residual_checkpoint": file_hash(args.residual_checkpoint),
                "recordings": {str(path): json.loads((path.parent / "metadata.json").read_text())
                               for path in paths if (path.parent / "metadata.json").is_file()},
                "validation_action_donors": action_donors(validation, val_indices).tolist(),
                "sources": {str(p): file_hash(p) for p in sources}, "motion_prior": file_hash(args.motion_baseline),
                "train_indices": indices.tolist(), "validation_indices": val_indices.tolist(),
                "epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size,
                "learning_rate": args.learning_rate, "rollout_steps": 2, "detach_rollout_inputs": True,
                "features": "Unwarped full384, train-only normalization; no PCA or goal fields in predictor",
                "training_targets": "Last-context predictions at t+2 and t+3; evaluated recursively at t+2 and t+6",
                "torch": str(torch.__version__), "device": args.device,
                "hardware": torch.cuda.get_device_name() if args.device == "cuda" else "cpu"}
    normalization = {"mean": mean, "scale": scale}
    reports = [train_arm(args, architecture, seed, train, validation, indices, val_indices, normalization, prior, manifest)
               for seed in args.seeds for architecture in ("dino", "adaln")]
    report = {"arms": reports, "scope": "Development-only offline pilot. No reserved test data or navigation promotion."}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "FINISHED").touch()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
