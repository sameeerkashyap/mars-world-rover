"""Prepare immutable planner comparisons and verify their native evidence."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = "815f19e8858bd47361a868fb7500e52d2449e76f"
FILES = {
    "scripts/rover_plan.py": "scripts/rover_plan.py",
    "rover_navigate.py": "deploy/nautilus-navigation/rover_navigate.py",
    "scripts/train_rover_baseline.py": "scripts/train_rover_baseline.py",
    "scripts/train_rover_jepa.py": "scripts/train_rover_jepa.py",
    "src/rover_forecast.py": "src/rover_forecast.py",
    "src/terrain_cost.py": "src/terrain_cost.py",
    "src/evaluation_terrain.py": "src/evaluation_terrain.py",
    "verify_navigation.py": "deploy/nautilus-jepa-navigation/verify.py",
    "validation.py": "deploy/nautilus-controller-validation/validation.py",
    "setup.sh": "deploy/nautilus-jepa-navigation/setup.sh",
    "runtime-setup.sh": "deploy/nautilus/runtime-setup.sh",
    "upstream.json": "notes/upstream.json",
}
SOURCE_KEYS = {"rover_navigate.py": "runner_source", "scripts/train_rover_baseline.py": "motion_source",
               "scripts/train_rover_jepa.py": "jepa_source", "src/rover_forecast.py": "forecast_source",
               "src/terrain_cost.py": "readout_source", "src/evaluation_terrain.py": "terrain_source"}
PRIMARY = {"pairs": 20, "minimum_success_gain": .1, "two_sided_alpha": .05,
           "maximum_mean_planning_s": .2, "no_added_collision_per_pair": True}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def design(phase, mode):
    if phase not in ("development", "fresh") or mode not in ("transport", "jepa", "model-hold"):
        raise ValueError("Use a registered phase and one fixed controller mode")
    cases = {"jepa": [(613, 112), (601, 100)],
             "transport": [(604, 103), (611, 110), (614, 113), (615, 114), (601, 100)],
             "model-hold": [(501, 30), (501, 31), (501, 32)]}[mode]
    return {"schema": "rover-controller-validation-v1", "phase": phase, "controllers": [mode],
            "development": [{"terrain": f"test-rocks-{layout}", "seeds": [seed]} for layout, seed in cases] if phase == "development" else [],
            "fresh": [{"terrain": f"test-rocks-{701 + i}", "seeds": [200 + i]} for i in range(20)] if phase == "fresh" else [],
            "steps": 750, "counterbalance_order": False, "audit_forecasts": False,
            "planner_order": ["original", "corrected"], "primary_test": PRIMARY if phase == "fresh" else None,
            "scope": "DEVELOPMENT regression evidence only" if phase == "development" else "Fixed original-versus-corrected native success endpoint"}


def validate_plan(plan):
    expected = design(plan["phase"], plan["controllers"][0])
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Registered design changed")
    if (plan["planners"]["original"]["commit"] != ORIGINAL
            or plan["planners"]["original"]["sha256"] != "0c12c0c42059a972ad7b67bff16f2831369291e90d2cc456fa19cb8e44d73415"
            or plan["sha256"]["motion_checkpoint"] != "76953ac81e383a6000d9b26bf8026b5049f4ce768831bd48506dea31d7649283"):
        raise ValueError("Original planner or motion weights changed")


def prepare(reference, output, phase, mode, checkpoint, predictor):
    def git_bytes(name):
        return subprocess.run(["rtk", "proxy", "git", "show", f"{reference}:{name}"], cwd=ROOT,
                              check=True, capture_output=True).stdout
    reference = subprocess.run(["rtk", "proxy", "git", "rev-parse", reference], cwd=ROOT,
                               check=True, capture_output=True, text=True).stdout.strip()
    plan = json.loads(git_bytes("deploy/nautilus-jepa-confirmation/frozen-v1-test-plan.json"))
    plan.update(design(phase, mode))
    plan["fresh_layout"]["seeds"] = [int(case["terrain"].removeprefix("test-rocks-")) for case in plan[phase]]
    plan["planners"] = {}
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for target, source in FILES.items():
        path = output / target
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(git_bytes(source))
        files[target] = digest(path)
    for role, commit in (("original", ORIGINAL), ("corrected", reference)):
        source = subprocess.run(["rtk", "proxy", "git", "show", f"{commit}:scripts/rover_plan.py"], cwd=ROOT,
                                check=True, capture_output=True).stdout
        target = f"planners/{role}/rover_plan.py"
        path = output / target
        path.parent.mkdir(parents=True)
        path.write_bytes(source)
        files[target] = digest(path)
        plan["planners"][role] = {"commit": commit, "path": target, "sha256": files[target]}
    for name, path in (("model.pt", checkpoint), ("meta_vit.py", predictor)):
        (output / name).write_bytes(path.read_bytes())
        files[name] = digest(output / name)
    if files["meta_vit.py"] != plan["sha256"]["predictor_source"]:
        raise ValueError("Use the frozen JEPA predictor source")
    plan["sha256"].update({key: files[name] for name, key in SOURCE_KEYS.items()})
    plan["sha256"].update(motion_checkpoint=files["model.pt"], predictor_source=files["meta_vit.py"],
                           planner_source=plan["planners"]["corrected"]["sha256"])
    plan["source_commit"] = reference
    plan["all_source_sha256"] = files
    validate_plan(plan)
    (output / "test-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    checksums = {**files, "test-plan.json": digest(output / "test-plan.json")}
    (output / "SHA256SUMS").write_text("".join(f"{value}  {name}\n" for name, value in sorted(checksums.items())))
    for path in output.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    return plan


def check_sources(snapshot, plan):
    validate_plan(plan)
    for name, expected in plan["all_source_sha256"].items():
        if digest(snapshot / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")


def run(snapshot, output, receipt, jepa_run, terrain_checkpoint, encoder_source):
    plan = json.loads((snapshot / "test-plan.json").read_text())
    check_sources(snapshot, plan)
    output.mkdir(parents=True, exist_ok=False)
    (output / "runtime.json").write_bytes(receipt.read_bytes())
    (output / "test-plan.json").write_bytes((snapshot / "test-plan.json").read_bytes())
    environment = {**os.environ, "PYTHONPATH": str(snapshot.resolve()) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    order = []
    for index, case in enumerate(plan[plan["phase"]]):
        for role in plan["planner_order"][::(-1 if index % 2 else 1)]:
            folder = output / f"{case['terrain']}-{case['seeds'][0]}-{role}"
            command = [sys.executable, str(snapshot / "rover_navigate.py"), "--terrain", case["terrain"],
                       "--seeds", str(case["seeds"][0]), "--steps", str(plan["steps"]), "--phase", plan["phase"],
                       "--checkpoint", str(snapshot / "model.pt"), "--output", str(folder),
                       "--planner-role", role, "--planner-source", str(snapshot / plan["planners"][role]["path"]),
                       "--runtime-receipt", str(output / "runtime.json"), "--record-decisions", "--viz", "none"]
            if plan["controllers"] == ["model-hold"]:
                command += ["--replay-plan", str(snapshot / "test-plan.json")]
            else:
                if not jepa_run or not terrain_checkpoint:
                    raise ValueError("Forecast comparisons require the frozen JEPA run and terrain checkpoint")
                command += ["--test-plan", str(snapshot / "test-plan.json"), "--jepa-run", str(jepa_run),
                            "--terrain-checkpoint", str(terrain_checkpoint), "--encoder-source", str(encoder_source),
                            "--predictor-source", str(snapshot / "meta_vit.py")]
            order.append(folder.name)
            (output / "execution-order.json").write_text(json.dumps(order, indent=2) + "\n")
            subprocess.run(command, check=True, env=environment)


def paired_result(pairs):
    if len(pairs) != 20:
        raise ValueError("All twenty fixed layout pairs are required")
    wins = sum(new["verified_success"] and not old["verified_success"] for old, new in pairs)
    losses = sum(old["verified_success"] and not new["verified_success"] for old, new in pairs)
    n = wins + losses
    probability = min(1., 2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)) / 2 ** n)
    gain = (wins - losses) / 20
    collisions = sum(new["collision"] and not old["collision"] for old, new in pairs)
    latency = sum(new["planning_mean_s"] for _, new in pairs) / 20
    return {"pairs": 20, "corrected_only_successes": wins, "original_only_successes": losses,
            "success_rate_gain": gain, "exact_two_sided_p": probability, "added_collision_pairs": collisions,
            "corrected_mean_planning_s": latency,
            "primary_passed": gain >= .1 and probability <= .05 and collisions == 0 and latency <= .2}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify(snapshot, output, verify_trial):
    plan = json.loads((snapshot / "test-plan.json").read_text())
    check_sources(snapshot, plan)
    if json.loads((output / "test-plan.json").read_text()) != plan:
        raise ValueError("Run plan differs from the source snapshot")
    terrain = load_module("comparison_terrain", snapshot / "src/evaluation_terrain.py")
    records, pairs, expected_order = [], [], []
    for index, case in enumerate(plan[plan["phase"]]):
        seed = case["seeds"][0]
        verified, starts, scans, environments = {}, [], [], []
        expected_meshes = terrain.heldout_meshes(int(case["terrain"].removeprefix("test-rocks-")))
        for role in plan["planner_order"][::(-1 if index % 2 else 1)]:
            name = f"{case['terrain']}-{seed}-{role}"
            expected_order.append(name)
            paths = list((output / name).glob("*/summary.json"))
            if len(paths) != 1:
                raise ValueError("Each role needs exactly one native summary")
            folder = paths[0].parent
            summary = json.loads(paths[0].read_text())
            config = summary["config"]
            if (config["frozen_test_plan"] != plan or config["planner_role"] != role
                    or config["terrain"] != case["terrain"] or config["seeds"] != [seed]
                    or config["controllers"] != plan["controllers"] or config["phase"] != plan["phase"]
                    or config["max_steps"] != 750 or config["success_distance_m"] != .18
                    or config["success_heading_rad"] != .1 or config["settling_steps"] != 5
                    or config["checkpoint_sha256"] != plan["sha256"]["motion_checkpoint"]
                    or config["runtime_sha256"] != digest(output / "runtime.json")
                    or digest(folder / "runtime.json") != digest(output / "runtime.json")
                    or config["planner_source_sha256"] != plan["planners"][role]["sha256"]
                    or len(summary["trials"]) != 1):
                raise ValueError("Executed comparison settings changed")
            sources = {Path(name).name: plan["all_source_sha256"][name] for name in SOURCE_KEYS
                       if name not in ("scripts/train_rover_jepa.py", "src/terrain_cost.py")}
            sources["rover_plan.py"] = plan["planners"][role]["sha256"]
            for name, expected in sources.items():
                if config["source_sha256"].get(name) != expected or digest(folder / "source" / name) != expected:
                    raise ValueError(f"Executed source changed: {name}")
            with np.load(folder / "terrain/meshes.npz") as meshes:
                for key, expected in zip(("ground", "faces", "rocks", "rock_faces"), expected_meshes):
                    if not np.array_equal(meshes[key], expected):
                        raise ValueError("Executed terrain mesh changed")
            environments.append({key: config[key] for key in ("task", "torch", "numpy", "cameras", "reset_joint_state")})
            trial = summary["trials"][0]
            if trial["seed"] != seed or trial["controller"] != plan["controllers"][0] or not trial["hold_reached_goal"]:
                raise ValueError("Trial mode or seed changed")
            starts.append(trial["initial"])
            with np.load(folder / f"seed-{seed}-{trial['controller']}-scan.npz") as arrays:
                scans.append(arrays["initial"].copy())
            verified[role] = {"role": role, "terrain": case["terrain"], **verify_trial(folder, trial)}
            records.append(verified[role])
        if environments[0] != environments[1]:
            raise ValueError("Paired simulator environment changed")
        for field, tolerance in (("position", .01), ("goal", 1e-5), ("yaw", .01), ("heading_error_rad", 1e-5),
                                 ("root_velocity", 1e-5), ("joint_position", 1e-5), ("joint_velocity", 1e-5)):
            if not np.allclose(starts[0][field], starts[1][field], atol=tolerance, rtol=0):
                raise ValueError(f"Unmatched initial {field}")
        if scans[0].shape != (10201,) or not np.allclose(scans[0], scans[1], atol=1e-5, rtol=0, equal_nan=True):
            raise ValueError("Unmatched initial scan")
        pairs.append((verified["original"], verified["corrected"]))
    if len(list(output.glob("*/*/summary.json"))) != len(expected_order):
        raise ValueError("Unexpected comparison summaries")
    if json.loads((output / "execution-order.json").read_text()) != expected_order:
        raise ValueError("Planner execution order changed")
    return {"verified": True, "phase": plan["phase"], "trials": records, "pairs": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_args = sub.add_parser("prepare")
    prepare_args.add_argument("reference")
    prepare_args.add_argument("snapshot", type=Path)
    prepare_args.add_argument("--phase", choices=("development", "fresh"), default="development")
    prepare_args.add_argument("--mode", choices=("transport", "jepa", "model-hold"), default="transport")
    prepare_args.add_argument("--checkpoint", type=Path, required=True)
    prepare_args.add_argument("--predictor-source", type=Path, required=True)
    for command in ("run", "verify", "analyze"):
        child = sub.add_parser(command)
        child.add_argument("snapshot", type=Path)
        child.add_argument("output", type=Path)
        if command == "run":
            child.add_argument("--runtime-receipt", type=Path, required=True)
            child.add_argument("--jepa-run", type=Path)
            child.add_argument("--terrain-checkpoint", type=Path)
            child.add_argument("--encoder-source", type=Path, default=Path("/runtime/dinov2"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.reference, args.snapshot, args.phase, args.mode, args.checkpoint, args.predictor_source)
    elif args.command == "run":
        run(args.snapshot, args.output, args.runtime_receipt, args.jepa_run, args.terrain_checkpoint, args.encoder_source)
    else:
        common = load_module("comparison_trace", Path(__file__).with_name("verify_navigation.py") if Path(__file__).with_name("verify_navigation.py").is_file() else ROOT / "deploy/nautilus-jepa-navigation/verify.py")
        report = verify(args.snapshot, args.output, common.verify_trial)
        if args.command == "analyze":
            if report["phase"] != "fresh":
                raise ValueError("Development cases cannot support a fresh endpoint")
            report = paired_result(report["pairs"])
        (args.output / (args.command + ".json")).write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
