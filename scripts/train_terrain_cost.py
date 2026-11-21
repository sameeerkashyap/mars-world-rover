"""Train a terrain-cost readout on existing recordings; keep JEPA frozen."""

import argparse
import json
from pathlib import Path
import shutil

import torch
from torch.nn import functional as F

from scripts.train_rover_baseline import model_for
from scripts.train_rover_jepa import (
    RoverJEPA, encode_episodes, file_hash, prior_motion, rollout, sequence_arrays, warp_features,
)
from src.terrain_cost import TerrainCost, terrain_targets


def recording_split(training, validation, split, manifest):
    hashes = {}
    seen = set()
    for name, paths in (("train", training), ("validation", validation)):
        allowed = {manifest["files"][path] for path in split[name]}
        for path in paths:
            digest = file_hash(path)
            if digest in seen or digest not in allowed:
                raise ValueError(f"Duplicate or out-of-split {name} recording: {path}")
            seen.add(digest)
            hashes[str(path)] = digest
    return hashes


def recording_data(paths, encoder, checkpoint, device):
    episodes = encode_episodes(paths, encoder, device)
    sequences = [sequence for path in paths for sequence in sequence_arrays(path)]
    result = []
    for (name, features, states, actions), (source, observations, _) in zip(episodes, sequences, strict=True):
        if name != source or len(features) != len(observations):
            raise ValueError("Terrain labels and encoded observations are misaligned")
        latent = ((features - checkpoint["mean"]) / checkpoint["scale"] @ checkpoint["projection"]
                  / checkpoint["latent_scale"])
        labels = terrain_targets(torch.from_numpy(observations[:, 5:].copy()))
        result.append((name, latent, states, actions, labels))
    return result


def fit_readout(model, training, validation, epochs, device):
    features = torch.cat([episode[1] for episode in training]).to(device)
    targets = torch.cat([episode[4] for episode in training]).to(device)
    val_features = torch.cat([episode[1] for episode in validation]).to(device)
    val_targets = torch.cat([episode[4] for episode in validation]).to(device)
    if epochs < 1 or not torch.isfinite(features).all() or not torch.isfinite(targets).all():
        raise ValueError("Use finite training data and a positive epoch count")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    best, history = float("inf"), []
    for epoch in range(epochs):
        model.train()
        for batch in torch.randperm(len(features), device=device).split(32):
            loss = F.mse_loss(model(features[batch]), targets[batch])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite terrain training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            error = F.mse_loss(model(val_features), val_targets).item()
        if not torch.isfinite(torch.tensor(error)):
            raise RuntimeError("Non-finite terrain validation loss")
        history.append({"epoch": epoch + 1, "validation_mse": error})
        if error < best:
            best, selected = error, epoch + 1
            weights = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(weights)
    return {"best_epoch": selected, "measured_validation_mse": best, "history": history}, targets.mean(dim=0).cpu()


@torch.inference_mode()
def evaluate_forecasts(head, world_model, episodes, constant, device):
    head.eval()
    world_model.eval()
    totals = {name: {key: 0. for key in ("jepa_mse", "copy_mse", "transport_mse", "constant_mse")}
              for name in ("one_second", "five_seconds")}
    count = 0
    for _, features, states, actions, labels in episodes:
        for starts in torch.arange(len(features) - 6).split(16):
            context = torch.stack([features[start:start + 2] for start in starts]).to(device)
            state = torch.stack([states[start:start + 2] for start in starts]).to(device)
            commands = torch.stack([actions[start:start + 6] for start in starts]).to(device)
            forecasts, _ = rollout(world_model, context, state, commands, 5)
            transported, previous = context[:, 1], state[:, 1, :2]
            transport = {}
            for step in range(5):
                motion = prior_motion(world_model.motion_prior, previous, commands[:, step + 1])
                transported = warp_features(transported, motion)
                previous = commands[:, step + 1, -2:]
                transport[step + 1] = transported
            for horizon, name in ((1, "one_second"), (5, "five_seconds")):
                target = labels[starts + 1 + horizon].to(device)
                predictions = {"jepa_mse": head(forecasts[:, horizon - 1]),
                               "copy_mse": head(context[:, 1]), "transport_mse": head(transport[horizon]),
                               "constant_mse": constant.to(device).expand_as(target)}
                for key, prediction in predictions.items():
                    totals[name][key] += F.mse_loss(prediction, target).item() * len(starts)
            count += len(starts)
    if not count:
        raise ValueError("No complete forecast windows")
    return {"windows": count, **{name: {key: value / count for key, value in scores.items()}
                                for name, scores in totals.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed JEPA run with split.json and training/best.pt")
    parser.add_argument("--data", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-data", type=Path, nargs="+", required=True)
    parser.add_argument("--encoder-source", type=Path, required=True)
    parser.add_argument("--predictor-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(0)
    device = torch.device(args.device)
    development = json.loads((args.run / "training/metrics.json").read_text())
    if not development["quality_passed"]:
        raise ValueError("Terrain training requires a JEPA development quality pass")
    path = args.run / "training/best.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    manifest = checkpoint["manifest"]
    if manifest["model"] != "rover-dino-wm-v6":
        raise ValueError("Terrain training requires the rover-dino-wm-v6 feature schema")
    sources = {"/opt/train/scripts/train_rover_jepa.py": Path(__file__).with_name("train_rover_jepa.py"),
               "/opt/train/meta_vit.py": args.predictor_source}
    if any(file_hash(source) != manifest["source_hashes"][name] for name, source in sources.items()):
        raise ValueError("JEPA source differs from the selected checkpoint")
    split = json.loads((args.run / "split.json").read_text())
    files = recording_split(args.data, args.validation_data, split, manifest)
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = torch.hub.load(str(args.encoder_source), "dinov2_vits14", source="local", pretrained=True)
    weights = Path(torch.hub.get_dir()) / "checkpoints/dinov2_vits14_pretrain.pth"
    if file_hash(weights) != "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9":
        raise ValueError("The pretrained encoder weights changed")
    encoder = encoder.to(device).eval().requires_grad_(False)
    training = recording_data(args.data, encoder, checkpoint, device)
    validation = recording_data(args.validation_data, encoder, checkpoint, device)
    head = TerrainCost().to(device)
    metrics, constant = fit_readout(head, training, validation, args.epochs, device)
    world_model = RoverJEPA(args.predictor_source, motion_prior=model_for(5, 2)).to(device)
    world_model.load_state_dict(checkpoint["model"], strict=True)
    world_model.eval().requires_grad_(False)
    metrics["forecasts"] = evaluate_forecasts(head, world_model, validation, constant, device)
    metrics["constant_measured_validation_mse"] = F.mse_loss(
        constant.expand_as(torch.cat([ep[4] for ep in validation])), torch.cat([ep[4] for ep in validation])).item()
    report = {"schema": "rover-terrain-cost-v1", "jepa_checkpoint_sha256": file_hash(path), "files": files,
              "training_files": [str(path) for path in args.data],
              "validation_files": [str(path) for path in args.validation_data],
              "training_frames": sum(len(ep[1]) for ep in training),
              "validation_frames": sum(len(ep[1]) for ep in validation), "epochs": args.epochs,
              "encoder_weights_sha256": file_hash(weights), "device": args.device, "torch": str(torch.__version__),
              "scope": "Height-rule blocked fractions; offline development check, not navigation or physical safety",
              "metrics": metrics}
    torch.save({"model": head.state_dict(), "mean": checkpoint["mean"], "scale": checkpoint["scale"],
                "projection": checkpoint["projection"], "latent_scale": checkpoint["latent_scale"],
                "constant": constant, "report": report}, args.output / "best.pt")
    restored = TerrainCost().to(device).eval()
    restored.load_state_dict(torch.load(args.output / "best.pt", map_location=device, weights_only=True)["model"])
    with torch.inference_mode():
        sample = validation[0][1][:2].to(device)
        torch.testing.assert_close(restored(sample), head(sample), atol=0, rtol=0)
    for source in (Path(__file__), Path(__file__).resolve().parents[1] / "src/terrain_cost.py"):
        shutil.copyfile(source, args.output / source.name)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "FINISHED").touch()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
