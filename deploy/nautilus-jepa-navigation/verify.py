"""Independently recompute navigation outcomes from terminal-safe CSV traces."""

import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def verify_trial(folder, trial):
    path = folder / f"seed-{trial['seed']}-{trial['controller']}.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    previous = trial["initial"]["position"][:2]
    target_heading = trial["initial"]["yaw"] + trial["initial"]["heading_error_rad"]
    distance = path_length = 0.
    collisions = ready = consecutive = longest = 0
    geometric_steps = flag_disagreements = 0
    goal_holds = 0
    for index, row in enumerate(rows):
        values = [float(row[key]) for key in ("x_m", "y_m", "yaw_rad", "distance_m", "heading_error_rad", "planning_s")]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Non-finite trace: {path}")
        position = [float(row["x_m"]), float(row["y_m"])]
        goal = [float(row["goal_x_m"]), float(row["goal_y_m"])]
        distance = math.dist(position, goal)
        heading = (target_heading - float(row["yaw_rad"]) + math.pi) % (2 * math.pi) - math.pi
        if (math.dist(goal, trial["initial"]["goal"][:2]) > 1e-5
                or abs(distance - float(row["distance_m"])) > 1e-5
                or abs(heading - float(row["heading_error_rad"])) > 1e-5
                or abs(float(row["speed_command"])) > 0.700001
                or abs(float(row["turn_command"])) > 0.350001
                or int(row["step"]) != index + 1 or abs(float(row["time_s"]) - (index + 1) * 0.2) > 1e-5
                or float(row["planning_s"]) < 0):
            raise ValueError(f"Invalid measurement or command: {path}")
        path_length += math.dist(previous, position)
        previous = position
        geometric = distance < 0.18 and abs(heading) < 0.1
        geometric_steps += geometric
        flag_disagreements += (row["native_success"] == "True") != geometric
        collisions += row["collision"] == "True"
        ready += row["forecast_ready"] == "True"
        stopped = float(row["speed_command"]) == float(row["turn_command"]) == 0
        holding = row.get("goal_hold") == "True"
        enabled = trial.get("hold_reached_goal", trial["controller"] == "model-hold")
        if holding and (not enabled or trial["controller"] == "forward" or not stopped):
            raise ValueError(f"Invalid goal-hold command: {path}")
        goal_holds += holding
        consecutive = consecutive + 1 if stopped else 0
        longest = max(longest, consecutive)
    if not rows:
        raise ValueError(f"Empty navigation trial: {path}")
    verified = rows[-1]["native_success"] == "True" and collisions == 0 and distance < 0.18 and abs(heading) < 0.1
    if (verified != trial["verified_success"] or bool(collisions) != trial["collision"]
            or len(rows) != trial["steps"] or abs(path_length - trial["path_length_m"]) > 1e-5
            or abs(distance - trial["final_distance_m"]) > 1e-5
            or longest != trial["max_consecutive_stop_steps"] or ready != trial["forecast_ready_steps"]):
        raise ValueError(f"Trial summary disagrees with raw measurements: {path}")
    if trial["controller"] in ("transport", "jepa") and len(rows) > 6 and not ready:
        raise ValueError(f"Forecast history never became ready: {path}")
    if "goal_hold_steps" in trial and trial["goal_hold_steps"] != goal_holds:
        raise ValueError(f"Goal-hold count disagrees with raw commands: {path}")
    return {"seed": trial["seed"], "controller": trial["controller"], "verified_success": verified,
            "collision": bool(collisions), "path_length_m": path_length, "final_distance_m": distance,
            "stalled": longest >= 25, "forecast_scoring_calls": trial["forecast_scoring_calls"],
            "geometric_goal_steps": geometric_steps, "native_geometry_disagreement_steps": flag_disagreements,
            "goal_hold_steps": goal_holds,
            "planning_mean_s": sum(float(row["planning_s"]) for row in rows) / len(rows)}


def verify(root):
    results = []
    plans = []
    for path in sorted(root.glob("*/*/summary.json")):
        summary = json.loads(path.read_text())
        config = summary["config"]
        plan = config["frozen_test_plan"]
        plans.append((config["phase"], plan))
        cases = {"terrain": config["terrain"], "seeds": config["seeds"]}
        if cases not in plan[config["phase"]] or config["controllers"] != ["model", "transport", "jepa"]:
            raise ValueError("Evaluation differs from the frozen plan")
        if len(summary["trials"]) != len(config["seeds"]) * 3:
            raise ValueError("Missing controller trials")
        for seed in config["seeds"]:
            trials = [trial for trial in summary["trials"] if trial["seed"] == seed]
            if {trial["controller"] for trial in trials} != set(config["controllers"]):
                raise ValueError("Unpaired controllers")
            for trial in trials:
                if (math.dist(trial["initial"]["position"], trials[0]["initial"]["position"]) > 0.01
                        or math.dist(trial["initial"]["goal"], trials[0]["initial"]["goal"]) > 1e-5
                        or abs(trial["initial"]["yaw"] - trials[0]["initial"]["yaw"]) > 0.01):
                    raise ValueError("Controllers did not start from matched states")
                results.append({"phase": config["phase"], "terrain": config["terrain"],
                                **verify_trial(path.parent, trial)})
    development = [plan for phase, plan in plans if phase == "development"]
    fresh = [plan for phase, plan in plans if phase == "fresh"]
    if (len(development) != 2 or len(fresh) != 2 or development[0] != development[1]
            or fresh[0] != fresh[1]):
        raise ValueError("Both complete case sets must use a frozen plan within each phase")
    if development[0] != fresh[0]:
        original_hash = hashlib.sha256(json.dumps(development[0], sort_keys=True).encode()).hexdigest()
        controller_keys = set(development[0]["sha256"]) - {"terrain_source"}
        setting_keys = ("controllers", "steps", "forecast_weight", "forecast_seconds", "control_dt_seconds", "development", "fresh")
        if (original_hash != fresh[0].get("development_plan_sha256")
                or any(development[0]["sha256"][key] != fresh[0]["sha256"].get(key) for key in controller_keys)
                or any(development[0][key] != fresh[0][key] for key in setting_keys)):
            raise ValueError("Retry changed the frozen controllers, trial cases, or scoring settings")
    report = {"verified": True, "trials": results, "controllers": {}}
    for phase in ("development", "fresh"):
        report["controllers"][phase] = {}
        for controller in ("model", "transport", "jepa"):
            trials = [trial for trial in results if trial["phase"] == phase and trial["controller"] == controller]
            report["controllers"][phase][controller] = {
                "trials": len(trials), "successes": sum(trial["verified_success"] for trial in trials),
                "collisions": sum(trial["collision"] for trial in trials),
                "stalled_trials": sum(trial["stalled"] for trial in trials),
                "geometry_only_trials": sum(trial["geometric_goal_steps"] > 0 and not trial["verified_success"] for trial in trials),
                "native_geometry_disagreement_trials": sum(trial["native_geometry_disagreement_steps"] > 0 for trial in trials),
                "mean_path_length_m": sum(trial["path_length_m"] for trial in trials) / len(trials),
                "mean_planning_s": sum(trial["planning_mean_s"] for trial in trials) / len(trials)}
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    verify(Path(sys.argv[1]))
