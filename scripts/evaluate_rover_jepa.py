"""Score a frozen JEPA checkpoint on the two reserved rover recordings."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

import torch


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-source", type=Path, default=Path("/runtime/dinov2"))
    args = parser.parse_args()
    development = json.loads((args.run / "training/metrics.json").read_text())
    if not development["quality_passed"]:
        raise ValueError("Final-test evaluation requires the development quality target to pass first")
    checkpoint_path = args.run / "training/best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    manifest = checkpoint["manifest"]
    source = args.run / "source"
    for name, expected in manifest["source_hashes"].items():
        if sha256(source / name.removeprefix("/opt/train/")) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    seen = {Path(path).resolve() for path in manifest["files"]}
    cases = {}
    for path in args.data:
        metadata = json.loads((path.parent / "metadata.json").read_text())
        key = (metadata["terrain"], metadata["seed"])
        if (key not in (("mars", 20), ("debug", 21)) or metadata["steps"] != 3000
                or metadata["task"] != "AAURoverEnvSimple-v0" or path.resolve() in seen or key in cases
                or path.name != f"rover_seed_{key[1]}.hdf5"):
            raise ValueError(f"Not a unique reserved final-test recording: {path}")
        cases[key] = path
    if len(cases) != 2:
        raise ValueError("Both reserved terrains are required")
    encoder_weights = Path(torch.hub.get_dir()) / "checkpoints/dinov2_vits14_pretrain.pth"
    encoder_hash = sha256(encoder_weights)
    if encoder_hash != "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9":
        raise ValueError("The pretrained encoder weights changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Final evaluation requires CUDA")
    torch.set_num_threads(2)
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("frozen_rover_jepa", source / "scripts/train_rover_jepa.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    baseline_source = Path(frozen.model_for.__code__.co_filename)
    if baseline_source.read_bytes() != (source / "scripts/train_rover_baseline.py").read_bytes():
        raise ValueError("The frozen motion-model source was not loaded")
    device = torch.device("cuda")
    prior = frozen.model_for(5, 2).requires_grad_(False)
    model = frozen.RoverJEPA(source / "meta_vit.py", motion_prior=prior).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    encoder = torch.hub.load(str(args.encoder_source), "dinov2_vits14", source="local", pretrained=True)
    encoder = encoder.to(device).eval().requires_grad_(False)
    mean, scale = checkpoint["mean"], checkpoint["scale"]
    projection, latent_scale = checkpoint["projection"], checkpoint["latent_scale"]

    def score(episodes):
        data = frozen.windows(episodes, mean, scale, projection, latent_scale)
        return frozen.evaluate(model, data, device, projection, latent_scale, model.motion_prior)

    validation_path = next(Path(path) for path in manifest["files"]
                           if "mars-rover-variation-1/rover_seed_1.hdf5" in path)
    if sha256(validation_path) != manifest["files"][str(validation_path)]:
        raise ValueError("The development validation recording changed")
    replay = score(frozen.encode_episodes([validation_path], encoder, device))
    for name in ("one_step_mse", "five_step_mse", "goal_one_step_mse", "goal_five_step_mse"):
        if abs(replay[name] - checkpoint["validation"][name]) > 1e-4:
            raise RuntimeError(f"Validation replay changed {name}")
    if not frozen.quality_passed(replay):
        raise RuntimeError("The selected checkpoint no longer passes development validation")
    # Open final observations only after source, encoder, and development replay checks pass.
    final_episodes, results = [], {}
    for (terrain, seed), path in sorted(cases.items()):
        episodes = frozen.encode_episodes([path], encoder, device)
        final_episodes.extend(episodes)
        results[f"{terrain}-{seed}"] = {"path": str(path), "sha256": sha256(path), "scores": score(episodes)}
    pooled = score(final_episodes)
    report = {"selected_run": str(args.run), "model": manifest["model"],
              "checkpoint_sha256": sha256(checkpoint_path), "encoder_weights_sha256": encoder_hash,
              "source_sha256": manifest["source_hashes"], "validation_replay": replay,
              "cases": results, "pooled": pooled, "quality_passed": frozen.quality_passed(pooled),
              "scope": "Offline prediction on two existing terrain layouts; not navigation or safety evidence"}
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(__file__, args.output / Path(__file__).name)
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "FINISHED").touch()
    if report["quality_passed"]:
        (args.output / "QUALITY_PASSED").touch()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
