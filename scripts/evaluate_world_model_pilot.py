"""Evaluate frozen pilot checkpoints with cross-episode actions and the current model control."""

import argparse
import json
from pathlib import Path

import torch

from scripts.train_rover_baseline import model_for
from scripts.train_rover_jepa import encode_episodes, file_hash
from scripts.train_world_model_pilot import action_donors, candidate_passed, evaluate, load_residual
from src.rover_training import WindowDataset, cached_episodes
from src.rover_world_model import RoverWorldModel


def validate_recordings(manifest, paths, split):
    names = manifest.get("validation_files", split.get("validation", []))
    expected = {manifest["files"][name] for name in names}
    actual = [file_hash(path) for path in paths]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError("Evaluation inputs must exactly match the frozen validation recordings")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", type=Path)
    parser.add_argument("--validation-data", type=Path, nargs="+", required=True)
    parser.add_argument("--encoder-source", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--motion-baseline", type=Path, required=True)
    parser.add_argument("--residual-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Use distinct nonnegative seeds")
    torch.set_num_threads(2)
    device = torch.device(args.device)
    split_path = args.split or args.run.parent / "split.json"
    split = json.loads(split_path.read_text()) if split_path.is_file() else {"validation": []}
    checkpoints = [args.run / f"{architecture}-seed-{seed}/best.pt" for seed in args.seeds for architecture in ("dino", "adaln")]
    first = torch.load(checkpoints[0], map_location="cpu", weights_only=True)
    validate_recordings(first["manifest"], args.validation_data, split)
    if file_hash(args.motion_baseline) != first["manifest"]["motion_prior"]:
        raise ValueError("Evaluation motion prior differs from the frozen training control")
    args.output.mkdir(parents=True, exist_ok=False)
    encoder = torch.hub.load(str(args.encoder_source), "dinov2_vits14", source="local", pretrained=True).to(device).eval().requires_grad_(False)
    encoder_hash = file_hash(Path(torch.hub.get_dir()) / "checkpoints/dinov2_vits14_pretrain.pth")
    if encoder_hash != first["manifest"]["encoder_weights"]:
        raise ValueError("Evaluation encoder differs from the frozen training encoder")
    signature = {"weights": encoder_hash, "preprocessing": file_hash(Path(encode_episodes.__code__.co_filename)),
                 "loader": file_hash(Path(model_for.__code__.co_filename)),
                 "encoder_source": file_hash(args.encoder_source / "hubconf.py")}
    episodes = cached_episodes(args.validation_data, encoder, device, args.cache_dir, signature, encode_episodes, file_hash)
    del encoder
    prior = model_for(5, 2).to(device).eval().requires_grad_(False)
    prior.load_state_dict(torch.load(args.motion_baseline, map_location=device, weights_only=True)["model"])
    residual = load_residual(args.residual_checkpoint, args.upstream, device)
    recordings = first["manifest"].get("recordings") or {
        str(path): json.loads((path.parent / "metadata.json").read_text())
        for path in args.validation_data if (path.parent / "metadata.json").is_file()}
    reports = []
    for path in checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        manifest = checkpoint["manifest"]
        if path.parent.name != f"{manifest['architecture']}-seed-{manifest['seed']}":
            raise ValueError("Checkpoint architecture or seed does not match its arm")
        validate_recordings(manifest, args.validation_data, split)
        if (manifest["validation_indices"] != first["manifest"]["validation_indices"]
                or manifest["motion_prior"] != first["manifest"]["motion_prior"]
                or manifest["encoder_weights"] != encoder_hash
                or any(not torch.equal(checkpoint["normalization"][key], first["normalization"][key]) for key in ("mean", "scale"))):
            raise ValueError("Comparison arms differ in validation inputs, normalization, or controls")
        norm = {key: value.to(device) for key, value in checkpoint["normalization"].items()}
        dataset = WindowDataset(episodes, checkpoint["normalization"]["mean"], checkpoint["normalization"]["scale"], None, None)
        model = RoverWorldModel(manifest["architecture"], args.upstream).to(device).eval().requires_grad_(False)
        model.load_state_dict(checkpoint["model"], strict=True)
        scores = evaluate(model, dataset, torch.tensor(manifest["validation_indices"]), norm["scale"], prior,
                          device, residual, norm, recordings)
        reports.append({"architecture": manifest["architecture"], "seed": manifest["seed"], "epoch": checkpoint["epoch"],
                        "checkpoint_sha256": file_hash(path), "validation": scores,
                        "candidate_gate": candidate_passed(scores)})
        print(json.dumps({"evaluated": str(path), "checkpoint_sha256": reports[-1]["checkpoint_sha256"]}), flush=True)
    sources = [Path(__file__), Path(evaluate.__code__.co_filename), Path(RoverWorldModel.forward.__code__.co_filename),
               Path(WindowDataset.batch.__code__.co_filename), Path(encode_episodes.__code__.co_filename),
               Path(model_for.__code__.co_filename), *sorted(args.upstream.rglob("*.py"))]
    report = {"schema": "rover-world-model-evaluation-v2", "arms": reports,
              "validation_indices": first["manifest"]["validation_indices"],
              "action_donor_indices": action_donors(dataset, torch.tensor(first["manifest"]["validation_indices"])).tolist(),
              "sources": {str(path): file_hash(path) for path in sources},
              "current_residual_checkpoint_sha256": file_hash(args.residual_checkpoint),
              "motion_prior_sha256": file_hash(args.motion_baseline), "encoder_weights_sha256": encoder_hash,
              "validation_files": {str(path): file_hash(path) for path in args.validation_data},
              "recordings": recordings,
              "split_sha256": file_hash(split_path) if split_path.is_file() else None,
              "scope": "Development re-evaluation of fixed selected weights; no retraining, test-set opening, or controller promotion."}
    (args.output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "FINISHED").touch()


if __name__ == "__main__":
    main()
