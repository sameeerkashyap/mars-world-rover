"""Diagnose global rock clearance using privileged meshes; never supply these paths to control."""

import argparse
import hashlib
import heapq
import json
import math
from pathlib import Path

import numpy as np
import torch

from scripts.rover_plan import scan_costmap
from src.evaluation_terrain import heldout_meshes


def distances(points, polygons):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    best = np.full(len(points), np.inf)
    nearest = np.full(len(points), -1)
    for index, polygon in enumerate(polygons):
        edges = np.roll(polygon, -1, axis=0) - polygon
        delta = points[:, None] - polygon[None]
        t = np.clip((delta * edges).sum(axis=2) / (edges * edges).sum(axis=1), 0, 1)
        d = np.linalg.norm(delta - t[:, :, None] * edges, axis=2).min(axis=1)
        cross = edges[None, :, 0] * delta[:, :, 1] - edges[None, :, 1] * delta[:, :, 0]
        inside = (cross >= -1e-9).all(axis=1) | (cross <= 1e-9).all(axis=1)
        d[inside] = 0
        improved = d < best
        best[improved] = d[improved]
        nearest[improved] = index
    return best, nearest


def synthetic_scan(ground, polygons, tops, center, yaw):
    axis = np.linspace(-2.5, 2.5, 101)
    x, y = np.meshgrid(axis, axis)
    world = np.stack((center[0] + np.cos(yaw) * x - np.sin(yaw) * y,
                      center[1] + np.sin(yaw) * x + np.cos(yaw) * y), axis=-1).reshape(-1, 2)
    grid = ground.reshape(401, 401, 3)
    coordinate = (world + 25) / .2
    if (coordinate < 0).any() or (coordinate >= 400).any():
        raise ValueError("Synthetic scan extends beyond the provided ground mesh")
    ix, iy = np.floor(coordinate).astype(int).T
    fx, fy = (coordinate - np.floor(coordinate)).T
    z00, z10 = grid[iy, ix, 2], grid[iy, ix + 1, 2]
    z01, z11 = grid[iy + 1, ix, 2], grid[iy + 1, ix + 1, 2]
    z = np.where(fx + fy <= 1, z00 + fx * (z10 - z00) + fy * (z01 - z00),
                 z11 + (1 - fx) * (z01 - z11) + (1 - fy) * (z10 - z11))
    for polygon, top in zip(polygons, tops):
        d, _ = distances(world, [polygon])
        inside = d == 0
        z[inside] = np.maximum(z[inside], top)
    return torch.from_numpy(-z.astype(np.float32))


def global_route(polygons, start, goal):
    if (np.concatenate((start, goal)) < -8).any() or (np.concatenate((start, goal)) > 38).any():
        raise ValueError("Global clearance probe covers only [-8,38] m in both axes")
    resolution = .1
    radius = math.sqrt(2) * (.7 + .05 + .05)
    segment_allowance = resolution * math.sqrt(2) / 2
    axis = np.arange(-8., 38.0001, resolution)
    x, y = np.meshgrid(axis, axis)
    points = np.stack((x, y), axis=-1).reshape(-1, 2)
    clearance, _ = distances(points, polygons)
    clearance = clearance.reshape(len(axis), len(axis))
    free = clearance > radius + segment_allowance
    targets = free & ((x - goal[0]) ** 2 + (y - goal[1]) ** 2 < .18 ** 2)
    source = tuple(np.round((start[::-1] + 8) / resolution).astype(int))
    def heuristic(point):
        return max(0., math.hypot(axis[point[1]] - goal[0], axis[point[0]] - goal[1]) - .18)
    pending = [(heuristic(source), 0., source)] if free[source] else []
    best = {source: 0.} if pending else {}
    parents = {}
    end = None
    while pending:
        _, traveled, point = heapq.heappop(pending)
        if traveled != best[point]:
            continue
        if targets[point]:
            end = point
            break
        row, col = point
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < len(axis) and 0 <= nc < len(axis)) or not free[nr, nc]:
                continue
            if dr and dc and (not free[row, nc] or not free[nr, col]):
                continue
            candidate = traveled + resolution * math.hypot(dr, dc)
            if candidate < best.get((nr, nc), math.inf):
                best[nr, nc] = candidate
                parents[nr, nc] = point
                heapq.heappush(pending, (candidate + heuristic((nr, nc)), candidate, (nr, nc)))
    result = {"grid_resolution_m": resolution, "grid_bounds_m": [-8., 38.],
              "all_heading_rock_clear_radius_m": radius, "segment_allowance_m": segment_allowance,
              "start_grid_free": bool(free[source]), "goal_disk_free_cells": int(targets.sum()),
              "route_found": end is not None, "visited_cells": len(best)}
    if end is not None:
        path = [end]
        while path[-1] != source:
            path.append(parents[path[-1]])
        path.reverse()
        xy = np.array([[axis[col], axis[row]] for row, col in path])
        centers = np.concatenate((start[None], xy))
        center_clearance, _ = distances(centers, polygons)
        maximum_half_segment = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).max()) / 2
        certified = float(center_clearance.min()) - maximum_half_segment
        assert certified > radius
        assert np.linalg.norm(xy[-1] - goal) < .18
        result.update(path_length_m=float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()),
                      path_xy_m=centers.tolist(), endpoint_goal_distance_m=float(np.linalg.norm(xy[-1] - goal)),
                      minimum_vertex_rock_distance_m=float(center_clearance.min()),
                      continuous_segment_rock_distance_lower_bound_m=certified)
    return result


def diagnose_trial(trace, receipts):
    summary_path = trace.with_suffix(".json")
    config_path = trace.parent / "config.json"
    summary = json.loads(summary_path.read_text())
    config = json.loads(config_path.read_text())
    layout = int(config["terrain"].removeprefix("test-rocks-"))
    initial = summary["initial"]
    goal = np.array(initial["goal"][:2], dtype=np.float64)
    start = np.array(initial["position"][:2], dtype=np.float64)
    theta = (initial["yaw"] + initial["heading_error_rad"] + math.pi) % (2 * math.pi) - math.pi
    if goal.shape != (2,) or start.shape != (2,) or not np.isfinite([*goal, *start, theta]).all():
        raise ValueError("Clearance diagnosis needs finite planar goal, start and target heading")
    arrays = heldout_meshes(layout)
    hashes = {name: hashlib.sha256(array.tobytes()).hexdigest()
              for name, array in zip(("ground", "faces", "rocks", "rock_faces"), arrays)}
    assert hashes == receipts[str(layout)], f"Generated mesh differs from the frozen receipt for {layout}"
    rocks = arrays[2].reshape(80, 8, 3)
    polygons = rocks[:, :4, :2].astype(np.float64)
    distance, nearest = distances([goal, start], polygons)
    result = {"layout": layout, "seed": summary["seed"], "controller": summary["controller"],
              "trace": str(trace), "sha256": {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
                                               for file in (trace, summary_path, config_path)},
              "mesh_sha256": hashes, "goal_xy_m": goal.tolist(), "start_xy_m": start.tolist(),
              "required_heading_rad": theta, "goal_to_rock_m": float(distance[0]),
              "nearest_goal_rock_index": int(nearest[0]), "start_to_rock_m": float(distance[1]),
              "all_goal_centers_rejected_by_inscribed_circle_bound": bool(distance[0] + .18 < .7),
              "goal_center_square_clear_all_headings": bool(distance[0] > .7 * math.sqrt(2)),
              "entire_goal_disk_square_clear_all_headings": bool(distance[0] - .18 > .7 * math.sqrt(2)),
              "synthetic_goal_scans": []}
    for offset in (-.099, 0., .099):
        scan = synthetic_scan(arrays[0], polygons, rocks[:, 4, 2], goal, theta + offset)
        costmap = scan_costmap(scan)
        result["synthetic_goal_scans"].append({"heading_offset_rad": offset, "center_cost": float(costmap[50, 50]),
                                               "center_free": bool(costmap[50, 50] < 100)})
    result["global_route"] = global_route(polygons, start, goal)
    if result["global_route"]["route_found"]:
        endpoint = np.array(result["global_route"]["path_xy_m"][-1])
        scan = synthetic_scan(arrays[0], polygons, rocks[:, 4, 2], endpoint, theta)
        result["global_route"]["endpoint_required_heading_scan_center_cost"] = float(scan_costmap(scan)[50, 50])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", type=Path, nargs="+")
    parser.add_argument("--mesh-receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    square = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
    assert np.allclose(distances([[0., 0.], [2., 0.], [2., 2.]], [square])[0], [0., 1., math.sqrt(2)])
    receipts = json.loads(args.mesh_receipts.read_text())
    report = {"schema": "rover-global-clearance-diagnosis-v1", "diagnostic_only": True,
              "geometry_source_sha256": hashlib.sha256(Path(heldout_meshes.__code__.co_filename).read_bytes()).hexdigest(),
              "planner_source_sha256": hashlib.sha256(Path(scan_costmap.__wrapped__.__code__.co_filename).read_bytes()).hexdigest(),
              "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "mesh_receipts_sha256": hashlib.sha256(args.mesh_receipts.read_bytes()).hexdigest(),
              "limits": "Privileged geometry is diagnostic only. Planar clearance certificates do not prove "
                        "native execution, dynamics, tracking, sensor coverage or scan acceptance along the complete route. "
                        "Synthetic scans are scored only at goal centers and route endpoints. No grid route is inconclusive.",
              "cases": [diagnose_trial(trace, receipts) for trace in args.traces]}
    output = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()


