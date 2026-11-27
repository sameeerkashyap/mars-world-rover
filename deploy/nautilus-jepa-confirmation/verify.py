"""Verify every paired trace before testing a preregistered success gain."""

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np


def paired_result(pairs):
    if len(pairs) != 20:
        raise ValueError("The fixed test needs all twenty independent layout pairs")
    wins = sum(j["verified_success"] and not t["verified_success"] for t, j in pairs)
    losses = sum(t["verified_success"] and not j["verified_success"] for t, j in pairs)
    discordant = wins + losses
    probability = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2 ** discordant)
    gain = (wins - losses) / len(pairs)
    added_collisions = sum(j["collision"] and not t["collision"] for t, j in pairs)
    latency = sum(j["planning_mean_s"] for _, j in pairs) / len(pairs)
    passed = gain >= 0.1 and probability <= 0.05 and not added_collisions and latency <= 0.2
    return {"pairs": len(pairs), "jepa_only_successes": wins, "transport_only_successes": losses,
            "success_rate_gain": gain, "exact_two_sided_p": probability,
            "added_collision_pairs": added_collisions, "jepa_mean_planning_s": latency,
            "promotion_passed": passed,
            "decision": "success_gain_supported" if passed else "navigation_benefit_not_established"}


def verify(root, verify_trial):
    plan = json.loads((root / "source/test-plan.json").read_text())
    if (plan["schema"] != "rover-jepa-confirmation-v1" or plan["controllers"] != ["transport", "jepa"]
            or plan["fresh"] != [{"terrain": f"test-rocks-{601 + i}", "seeds": [100 + i]} for i in range(20)]
            or not plan["audit_forecasts"] or not plan["counterbalance_order"]):
        raise ValueError("The frozen design changed")
    for name, expected in plan["all_source_sha256"].items():
        candidates = [path for path in (root / "source").rglob(name)
                      if not any(part.startswith(".") for part in path.relative_to(root / "source").parts)]
        if len(candidates) != 1 or hashlib.sha256(candidates[0].read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen experiment source changed: {name}")
    paths = sorted(root.glob("fresh-*/*/summary.json"))
    if len(paths) != 20:
        raise ValueError("All twenty layouts must finish before analysis")
    pairs, records, seen = [], [], set()
    for path in paths:
        summary = json.loads(path.read_text())
        config = summary["config"]
        case = {"terrain": config["terrain"], "seeds": config["seeds"]}
        seed = config["seeds"][0]
        order = ["jepa", "transport"] if seed % 2 else plan["controllers"]
        if (case not in plan["fresh"] or config["terrain"] in seen or config["frozen_test_plan"] != plan
                or config["phase"] != "fresh" or config["max_steps"] != 750
                or config["controllers"] != plan["controllers"] or len(summary["trials"]) != 2
                or [t["controller"] for t in summary["trials"]] != order
                or config["checkpoint_sha256"] != plan["sha256"]["motion_checkpoint"]
                or config["success_distance_m"] != 0.18 or config["success_heading_rad"] != 0.1):
            raise ValueError("The executed cases or controller settings changed")
        seen.add(config["terrain"])
        source_names = {"rover_navigate.py": "runner_source", "rover_plan.py": "planner_source",
                        "train_rover_baseline.py": "motion_source", "rover_forecast.py": "forecast_source",
                        "evaluation_terrain.py": "terrain_source"}
        for name, key in source_names.items():
            if (config["source_sha256"][name] != plan["sha256"][key]
                    or hashlib.sha256((path.parent / "source" / name).read_bytes()).hexdigest() != plan["sha256"][key]):
                raise ValueError(f"Executed source changed: {name}")
        trials = {trial["controller"]: trial for trial in summary["trials"]}
        baseline, learned = trials["transport"], trials["jepa"]
        for field, tolerance in (("position", 0.01), ("goal", 1e-5), ("yaw", 0.01)):
            if not np.allclose(baseline["initial"][field], learned["initial"][field], atol=tolerance, rtol=0):
                raise ValueError(f"Unmatched initial {field}")
        for field in ("heading_error_rad", "root_velocity", "joint_position", "joint_velocity"):
            if not np.allclose(baseline["initial"][field], learned["initial"][field], atol=1e-5, rtol=0):
                raise ValueError(f"Unmatched initial {field}")
        scans = []
        verified = {}
        for controller, trial in trials.items():
            if trial["seed"] != seed or not trial["hold_reached_goal"]:
                raise ValueError("Trial seed or goal hold changed")
            with np.load(path.parent / f"seed-{seed}-{controller}-scan.npz") as arrays:
                scans.append(arrays["initial"].copy())
            with (path.parent / f"seed-{seed}-{controller}.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            audited = [row for row in rows if row["alternate_speed_command"] != ""]
            disagreements = sum(float(row["alternate_speed_command"]) != float(row["speed_command"])
                                or float(row["alternate_turn_command"]) != float(row["turn_command"]) for row in audited)
            if (len(audited) != trial["forecast_comparison_steps"] or disagreements != trial["forecast_command_disagreements"]
                    or len(audited) != trial["forecast_scoring_calls"]):
                raise ValueError("Forecast influence audit is missing or inconsistent")
            verified[controller] = {"terrain": config["terrain"], **verify_trial(path.parent, trial),
                                    "forecast_comparison_steps": len(audited), "forecast_command_disagreements": disagreements}
            records.append(verified[controller])
        if not np.allclose(scans[0], scans[1], atol=1e-5, rtol=0, equal_nan=True):
            raise ValueError("Paired initial height scans differ")
        pairs.append((verified["transport"], verified["jepa"]))
    report = {"verified": True, "primary": paired_result(pairs), "controllers": {}, "trials": records}
    for controller in plan["controllers"]:
        trials = [trial for trial in records if trial["controller"] == controller]
        report["controllers"][controller] = {
            "successes": sum(t["verified_success"] for t in trials), "collisions": sum(t["collision"] for t in trials),
            "stalled_trials": sum(t["stalled"] for t in trials),
            "mean_path_length_m": sum(t["path_length_m"] for t in trials) / len(trials),
            "mean_planning_s": sum(t["planning_mean_s"] for t in trials) / len(trials),
            "forecast_command_disagreements": sum(t["forecast_command_disagreements"] for t in trials)}
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    common = Path(__file__).with_name("verify_navigation.py")
    if not common.is_file():
        common = Path(__file__).parents[1] / "nautilus-jepa-navigation/verify.py"
    spec = importlib.util.spec_from_file_location("verify_navigation", common)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    verify(Path(sys.argv[1]), module.verify_trial)
