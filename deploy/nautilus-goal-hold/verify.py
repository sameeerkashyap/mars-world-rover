"""Verify paired development replays without changing native success criteria."""

import importlib.util
import json
import math
from pathlib import Path
import sys


def verify(folder, verify_trial):
    summary = json.loads((folder / "summary.json").read_text())
    config = summary["config"]
    plan = config["frozen_test_plan"]
    if (not config["development_replay"] or config["phase"] != "development"
            or config["controllers"] != ["model", "model-hold"]
            or {"terrain": config["terrain"], "seeds": config["seeds"]} not in plan["development"]
            or config["checkpoint_sha256"] != plan["sha256"]["motion_checkpoint"]):
        raise ValueError("Replay differs from its frozen development plan")
    if len(summary["trials"]) != len(config["seeds"]) * 2:
        raise ValueError("Missing paired goal-hold trials")
    for filename, key in (("rover_navigate.py", "runner_source"), ("rover_plan.py", "planner_source"),
                          ("train_rover_baseline.py", "motion_source"), ("rover_forecast.py", "forecast_source"),
                          ("evaluation_terrain.py", "terrain_source")):
        if config["source_sha256"][filename] != plan["sha256"][key]:
            raise ValueError("Replay source differs from the frozen plan")
    results = []
    for seed in config["seeds"]:
        pair = [trial for trial in summary["trials"] if trial["seed"] == seed]
        if len(pair) != 2 or {trial["controller"] for trial in pair} != set(config["controllers"]):
            raise ValueError("Unpaired replay controllers")
        if (math.dist(pair[0]["initial"]["position"], pair[1]["initial"]["position"]) > 0.01
                or math.dist(pair[0]["initial"]["goal"], pair[1]["initial"]["goal"]) > 1e-5
                or abs(pair[0]["initial"]["yaw"] - pair[1]["initial"]["yaw"]) > 0.01):
            raise ValueError("Replay initial states differ")
        for trial in pair:
            result = verify_trial(folder, trial)
            results.append(result)
    report = {"verified": True, "trials": results, "controllers": {}}
    for controller in config["controllers"]:
        trials = [trial for trial in results if trial["controller"] == controller]
        report["controllers"][controller] = {
            "trials": len(trials), "successes": sum(trial["verified_success"] for trial in trials),
            "collisions": sum(trial["collision"] for trial in trials),
            "goal_hold_steps": sum(trial["goal_hold_steps"] for trial in trials),
            "geometry_only_trials": sum(trial["geometric_goal_steps"] > 0 and not trial["verified_success"] for trial in trials)}
    report["scope"] = "Known-layout development replay; no unseen-layout or learned-prediction claim"
    (folder / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    spec = importlib.util.spec_from_file_location("verify_navigation_traces", Path(__file__).with_name("verify_navigation.py"))
    traces = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(traces)
    verify(Path(sys.argv[1]), traces.verify_trial)
