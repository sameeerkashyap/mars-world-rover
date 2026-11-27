"""Inspect saved navigation traces and last pre-command scans without running physics."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.rover_plan import plan_action, scan_costmap, scan_waypoint
from scripts.train_rover_baseline import model_for, predict, relative_motion


def measured_route(costmap, goal):
    grid = costmap[::2, ::2].tolist()
    reached = set()
    pending = [(25, 25)] if grid[25][25] < 100 else []
    reached.update(pending)
    while pending:
        row, col = pending.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 51 and 0 <= nc < 51) or grid[nr][nc] >= 100 or (nr, nc) in reached:
                continue
            if dr and dc and (grid[row][nc] >= 100 or grid[nr][col] >= 100):
                continue
            reached.add((nr, nc))
            pending.append((nr, nc))
    gx, gy = goal.tolist()
    covered = abs(gx) <= 2.5 and abs(gy) <= 2.5
    goal_cell = (round(gy / .1) + 25, round(gx / .1) + 25)
    blocked = (costmap >= 100).nonzero()
    margin = float(((blocked - 50).float().norm(dim=1) * .05).min()) if len(blocked) else None
    waypoint = scan_waypoint(costmap, goal)
    return {"goal_xy_m": [gx, gy], "goal_in_scan": covered,
            "goal_reachable": goal_cell in reached if covered else None,
            "goal_cell_hard_blocked": grid[goal_cell[0]][goal_cell[1]] >= 100 if covered else None,
            "start_hard_blocked": grid[25][25] >= 100,
            "hard_blocked_fraction": float((costmap >= 100).float().mean()),
            "distance_to_hard_mask_m": margin, "footprint_half_width_m": .7, "reachable_cells": len(reached),
            "closest_reachable_goal_distance_m": min((math.hypot((c - 25) * .1 - gx, (r - 25) * .1 - gy)
                                                       for r, c in reached), default=None),
            "waypoint_m": waypoint.tolist() if waypoint is not None else None}


@torch.inference_mode()
def candidate_coverage(model, state, costmap, waypoint, dense=False):
    actions = torch.cartesian_prod(state.new_tensor([0., .05, .15, .35, .65, -.05, -.2]),
                                  state.new_tensor([0., -.35, -.23, -.12, -.04, .04, .12, .23, .35]))
    durations = list(range(1, 61)) if dense else [10, 20, 40, 60]
    turns = torch.cartesian_prod(state.new_tensor([.35, .65]), state.new_tensor([-.35, .35]),
                                state.new_tensor(durations))
    follow = torch.cat((actions, torch.stack((turns[:, 0], torch.zeros_like(turns[:, 0])), dim=1)))
    actions = torch.cat((actions, torch.stack((torch.zeros_like(turns[:, 1]), turns[:, 1]), dim=1)))
    turn_steps = torch.cat((torch.zeros(63, dtype=torch.long), turns[:, 2].long()))
    lengths = turn_steps + 20
    initial = state.expand(len(actions), -1).clone()
    prediction = initial.clone()
    risk = state.new_zeros(len(actions))
    for step in range(int(lengths.max())):
        command = torch.where((step < turn_steps)[:, None], actions, follow)
        prediction = torch.where((step < lengths)[:, None], predict(model, prediction, command), prediction)
        xy = relative_motion(initial, prediction)[:, :2] / .11
        index = ((xy + 2.5) / .05).round().long()
        outside = ((index < 0) | (index > 100)).any(dim=1)
        sampled = costmap[index[:, 1].clamp(0, 100), index[:, 0].clamp(0, 100)]
        risk = torch.maximum(risk, torch.where(outside, 100.5, sampled))
    error = (xy - waypoint).norm(dim=1)
    cost = error + risk
    safe = risk < 100
    cost[~safe] = torch.inf
    if not safe.any():
        return {"count": len(actions), "safe": 0}
    winner = int(cost.argmin())
    forward = safe & (actions[:, 0] > 0)
    return {"count": len(actions), "safe": int(safe.sum()), "safe_forward_first": int(forward.sum()),
            "best_first_action": actions[winner].tolist(), "best_follow_action": follow[winner].tolist(),
            "best_turn_steps": int(turn_steps[winner]), "best_endpoint_m": xy[winner].tolist(),
            "best_waypoint_error_m": float(error[winner]), "best_cost": float(cost[winner]),
            "best_forward_first_error_m": float(error[forward].min()) if forward.any() else None}


def trace_window(rows):
    tail = rows[-100:]
    xy = np.array([[float(row["x_m"]), float(row["y_m"])] for row in tail])
    yaw = np.unwrap([float(row["yaw_rad"]) for row in tail])
    turns = [np.sign(float(row["turn_command"])) for row in tail if abs(float(row["turn_command"])) > 1e-6]
    turn_only = [abs(float(row["speed_command"])) < 1e-6 and abs(float(row["turn_command"])) > 1e-6 for row in rows]
    longest = current = 0
    for turning in turn_only:
        current = current + 1 if turning else 0
        longest = max(current, longest)
    return {"steps": len(tail), "turn_only_steps": sum(turn_only[-100:]),
            "turn_sign_changes": int(sum(a != b for a, b in zip(turns, turns[1:]))),
            "yaw_range_rad": float(np.ptp(yaw)), "net_motion_m": float(np.linalg.norm(xy[-1] - xy[0])),
            "path_length_m": float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
            "distance_reduction_m": float(tail[0]["distance_m"]) - float(tail[-1]["distance_m"]),
            "longest_turn_only_run_all_steps": longest}


def diagnose_trial(path, model):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("Last-scan reconstruction needs at least two CSV samples")
    numeric = ("x_m", "y_m", "yaw_rad", "goal_x_m", "goal_y_m", "distance_m", "heading_error_rad",
               "speed_command", "turn_command")
    for step, row in enumerate(rows, 1):
        if int(row["step"]) != step or not all(math.isfinite(float(row[key])) for key in numeric):
            raise ValueError("Trace needs consecutive steps and finite state fields")
        distance = math.hypot(float(row["goal_x_m"]) - float(row["x_m"]), float(row["goal_y_m"]) - float(row["y_m"]))
        if not math.isclose(distance, float(row["distance_m"]), abs_tol=2e-5):
            raise ValueError("Trace distance does not match its pose and goal")
    summary = json.loads(path.with_suffix(".json").read_text())
    if summary["steps"] != len(rows):
        raise ValueError("Trace is incomplete for its trial summary")
    pre, final = rows[-2:]
    bearing = math.atan2(float(pre["goal_y_m"]) - float(pre["y_m"]), float(pre["goal_x_m"]) - float(pre["x_m"])) - float(pre["yaw_rad"])
    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
    state = torch.tensor([[float(pre["speed_command"]), float(pre["turn_command"]),
                           float(pre["distance_m"]) * .11, bearing / math.pi,
                           float(pre["heading_error_rad"]) / math.pi]])
    scan_path = path.with_name(path.stem + "-scan.npz")
    with np.load(scan_path) as scans:
        scan = torch.from_numpy(scans["last"].copy()).float()
    costmap = scan_costmap(scan)
    goal = state.new_tensor([math.cos(bearing), math.sin(bearing)]) * float(pre["distance_m"])
    route = measured_route(costmap, goal)
    align_heading = False
    for row in [summary.get("initial", rows[0]), *rows[:-1]]:
        distance, heading = float(row["distance_m"]), abs(float(row["heading_error_rad"]))
        if distance < .16:
            align_heading = True
        if distance > .25 or (distance > .18 and heading < .08):
            align_heading = False
    diagnostics = {}
    action = plan_action(model, state, height_scan=scan, align_heading=align_heading,
                         hold_reached_goal=summary.get("hold_reached_goal", True), diagnostics=diagnostics)
    plain_action = plan_action(model, state, height_scan=scan, align_heading=align_heading,
                               hold_reached_goal=summary.get("hold_reached_goal", True))
    if not torch.equal(action, plain_action):
        raise ValueError("Observational diagnostics changed the public planner action")
    candidates = {}
    if diagnostics["reason"] == "selected" and diagnostics["mode"] == "waypoint":
        waypoint = state.new_tensor(route["waypoint_m"])
        for name, dense in (("standard", False), ("dense_turn_duration", True)):
            candidates[name] = candidate_coverage(model, state, costmap, waypoint, dense)
        if candidates["standard"].get("safe", 0) and not np.allclose(action[0], candidates["standard"]["best_first_action"], atol=1e-6):
            raise ValueError("Candidate probe differs from the public planner; inspect changed planning rules")
    return {"trace": str(path), "sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                            for file in (path, path.with_suffix(".json"), scan_path)},
            "scan_pre_command_step": int(pre["step"]), "final_step": int(final["step"]),
            "state": state[0].tolist(), "native_timeout": summary["final"]["termination"]["time_limit"],
            "native_success": final["native_success"] == "True", "collision": final["collision"] == "True",
            "verified_success": final["native_success"] == "True" and final["collision"] == "False"
                                and float(final["distance_m"]) < .18 and abs(float(final["heading_error_rad"])) < .1,
            "planning_mode": "hold" if diagnostics["reason"] == "goal_hold" else diagnostics["mode"],
            "decision": diagnostics,
            "diagnostics_action_unchanged": True,
            "final_distance_m": float(final["distance_m"]), "route": route, "candidates": candidates,
            "scan_only_action": action[0].tolist(),
            "recorded_final_action": [float(final["speed_command"]), float(final["turn_command"])],
            "last_100_steps": trace_window(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != "goal-motion-v1" or not checkpoint.get("use_actions"):
        raise ValueError("Diagnosis requires the frozen goal-motion-v1 action model")
    model = model_for(5, 2)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    report = {"checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "planner_sha256": hashlib.sha256(Path(plan_action.__wrapped__.__code__.co_filename).read_bytes()).hexdigest(),
              "limits": "Only the last pre-command scan is saved. Scan-only scoring omits historical forecasts. "
                        "Reachability and candidates use the measured local mask and frozen learned motion, not physics. "
                        "A final scan cannot prove an earlier route, global reachability, repeated waypoint selection, "
                        "or that a longer committed turn is safe on future scans.",
              "trials": [diagnose_trial(path, model) for path in args.traces]}
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")
    if args.require_success and any(not trial["verified_success"] for trial in report["trials"]):
        raise SystemExit("Native timeout or failure remains in saved trace")


if __name__ == "__main__":
    main()
