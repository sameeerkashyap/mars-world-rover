"""Verify native pre-command captures and summarize route-switching evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.diagnose_rover_timeouts import measured_route
from scripts.rover_plan import scan_costmap


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def range_stats(values):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return {"min": float(finite.min()), "median": float(np.median(finite)), "max": float(finite.max())} if len(finite) else None


def fine_component(risk):
    grid = risk.tolist()
    pending = [(50, 50)] if grid[50][50] < 100 else []
    reached = set(pending)
    while pending:
        row, col = pending.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 101 and 0 <= nc < 101) or grid[nr][nc] >= 100 or (nr, nc) in reached:
                continue
            if dr and dc and (grid[row][nc] >= 100 or grid[nr][col] >= 100):
                continue
            reached.add((nr, nc))
            pending.append((nr, nc))
    return reached


def inspect_trial(path):
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    summary_path = path.with_suffix(".json")
    summary = json.loads(summary_path.read_text())
    config_path = path.parent / "config.json"
    config = json.loads(config_path.read_text())
    for name, checksum in config["source_sha256"].items():
        assert hashlib.sha256((path.parent / "source" / name).read_bytes()).hexdigest() == checksum
    hdf_path = path.with_name(path.stem + "-decisions.h5")
    with h5py.File(hdf_path) as handle:
        assert handle.attrs["schema"] == "rover-plan-decisions-v1" and handle.attrs["complete"]
        n = int(handle.attrs["steps_written"])
        assert n == len(rows) == summary["steps"] and n > 0
        policy = handle["policy"][:n]
        pose = handle["position_w"][:n].astype(float)
        yaw = handle["yaw_rad"][:n].astype(float)
        commands = handle["command"][:n].astype(float)
        details = [json.loads(value) for value in handle["diagnostics"][:n]]
    assert policy.shape == (n, 10206)
    with np.load(path.with_name(path.stem + "-scan.npz")) as scans:
        np.testing.assert_array_equal(policy[0, 5:], scans["initial"])
        np.testing.assert_array_equal(policy[-1, 5:], scans["last"])
    initial = summary["initial"]
    expected_pose = [initial["position"], *[[float(row[key]) for key in ("x_m", "y_m", "z_m")] for row in rows[:-1]]]
    expected_yaw = [initial["yaw"], *[float(row["yaw_rad"]) for row in rows[:-1]]]
    csv_commands = [[float(row[key]) for key in ("speed_command", "turn_command")] for row in rows]
    np.testing.assert_array_equal(commands, np.asarray(csv_commands, dtype=np.float32))
    np.testing.assert_array_equal(pose, np.asarray(expected_pose, dtype=np.float32))
    np.testing.assert_array_equal(yaw, np.asarray(expected_yaw, dtype=np.float32))
    np.testing.assert_array_equal(policy[:, :2], np.concatenate((np.zeros((1, 2)), commands[:-1])))
    state = policy[:, :5].astype(float)
    delta = np.asarray(initial["goal"][:2]) - pose[:, :2]
    errors = {"distance_m": state[:, 2] / .11 - np.linalg.norm(delta, axis=1),
              "bearing_rad": wrap(state[:, 3] * np.pi - wrap(np.arctan2(delta[:, 1], delta[:, 0]) - yaw)),
              "heading_rad": wrap(state[:, 4] * np.pi - wrap(initial["yaw"] + initial["heading_error_rad"] - yaw))}
    world = np.array([row["world_waypoint_m"] if row["world_waypoint_m"] is not None else [np.nan, np.nan] for row in details])
    for index, detail in enumerate(details):
        local = detail["waypoint_m"]
        if local is not None:
            c, s = np.cos(yaw[index]), np.sin(yaw[index])
            expected = pose[index, :2] + [c * local[0] - s * local[1], s * local[0] + c * local[1]]
            np.testing.assert_allclose(world[index], expected, rtol=0, atol=1e-6)
    jumps = np.linalg.norm(np.diff(world, axis=0), axis=1)
    reversals = np.flatnonzero(commands[1:, 1] * commands[:-1, 1] < 0) + 1
    turn_steps = np.array([row.get("turn_steps") or 0 for row in details])
    prefixes = np.flatnonzero(turn_steps > 0)
    eligible = [i for i in prefixes if i + turn_steps[i] < n]
    selected = [row for row in details if row["reason"] == "selected"]
    gaps = np.array([np.nan if row.get("top_cost_gap") is None else row["top_cost_gap"] for row in details])
    retention = dict(large_switches=0, old_point_hard_free=0, coarse_reachable=0, fine_reachable=0, coarse_target_hard_blocked=0)
    for index in range(max(1, n - 100), n):
        if not jumps[index - 1] > 1:
            continue
        delta = world[index - 1] - pose[index, :2]
        c, s = np.cos(yaw[index]), np.sin(yaw[index])
        local = torch.tensor([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]], dtype=torch.float32)
        risk = scan_costmap(torch.from_numpy(policy[index, 5:]))
        route = measured_route(risk, local)
        cell = ((local + 2.5) / .05).round().long()
        covered = bool(((cell >= 0) & (cell <= 100)).all())
        retention["large_switches"] += 1
        retention["old_point_hard_free"] += int(covered and float(risk[cell[1], cell[0]]) < 100)
        retention["coarse_reachable"] += int(route["goal_reachable"] is True)
        retention["coarse_target_hard_blocked"] += int(route["goal_cell_hard_blocked"] is True)
        retention["fine_reachable"] += int(covered and (int(cell[1]), int(cell[0])) in fine_component(risk))
    return {"trace": str(path), "sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                             for file in (path, summary_path, hdf_path, config_path)},
            "frozen_source_sha256": config["source_sha256"], "steps": n, "alignment": "exact",
            "native_success": summary["native_success"], "verified_success": summary["verified_success"],
            "final_termination": summary["final"]["termination"],
            "policy_error_max": {key: float(np.abs(value).max()) for key, value in errors.items()},
            "safe_count": range_stats([row["safe_count"] for row in selected]),
            "safe_forward_first_count": range_stats([row["safe_forward_first_count"] for row in selected]),
            "top_cost_gap": range_stats(gaps), "gap_at_reversals": range_stats(gaps[reversals]),
            "world_waypoint_jump_m": range_stats(jumps), "waypoint_jumps_over_1m": int(np.sum(jumps > 1)),
            "turn_reversals": len(reversals), "large_jumps_coincident_with_reversal": int(np.sum(jumps[reversals - 1] > 1)),
            "selected_turn_steps": {str(value): int(np.sum(turn_steps == value)) for value in np.unique(turn_steps)},
            "prefix_decisions": len(prefixes), "eligible_full_prefixes": len(eligible),
            "executed_identical_prefix": sum(bool(np.all(np.isclose(commands[i:i + turn_steps[i]], commands[i], atol=1e-6))) for i in eligible),
            "follow_command_at_prefix_horizon": sum(bool(np.allclose(commands[i + turn_steps[i]], details[i]["follow_action"], atol=1e-6)) for i in eligible),
            "last_100": {"turn_reversals": int(np.sum(reversals >= n - 100)),
                         "turn_only_steps": int(np.sum((commands[-100:, 0] == 0) & (commands[-100:, 1] != 0))),
                         "median_waypoint_jump_m": range_stats(jumps[-99:])["median"] if np.isfinite(jumps[-99:]).any() else None,
                         "net_motion_m": float(np.linalg.norm(pose[-1, :2] - pose[max(0, n - 100), :2])),
                         "prior_waypoint": retention}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    wall = torch.zeros((101, 101))
    wall[:, 60] = 100
    assert (50, 70) not in fine_component(wall)
    wall[51, 60] = 0
    assert (50, 70) in fine_component(wall)
    assert measured_route(wall, torch.tensor([1., 0.]))["goal_reachable"] is False
    report = {"schema": "rover-native-decision-diagnosis-v1",
              "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "diagnostic_sources_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                              for name in ("rover_plan.py", "diagnose_rover_timeouts.py")},
              "limits": "Known development evidence only. Large waypoint switches and prior-point reachability "
                        "support a retention hypothesis, not a causal fix. Fine-grid checks do not change production "
                        "routing. Per-switch details remain in the raw HDF. Pose alignment does not certify scan registration.",
              "trials": [inspect_trial(path) for path in args.traces]}
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
