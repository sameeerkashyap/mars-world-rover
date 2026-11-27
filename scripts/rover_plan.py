"""Choose a bounded command by rolling the learned rover model forward."""

import heapq
import math

import torch

from scripts.train_rover_baseline import predict, relative_motion


@torch.inference_mode()
def scan_costmap(scan):
    if scan.numel() != 101 * 101:
        raise ValueError("Clearance needs the configured 101 by 101 height scan")
    height = -scan.reshape(101, 101)
    finite = torch.isfinite(height)
    if not finite.any():
        return torch.full_like(height, 100.5)
    height = torch.where(finite, height, height[finite].median())
    axis = torch.linspace(-2.5, 2.5, 101, device=height.device, dtype=height.dtype)
    slope_x = (height[:, 1:] - height[:, :-1]).median() / 0.05
    slope_y = (height[1:, :] - height[:-1, :]).median() / 0.05
    level = height - slope_x * axis[None, :] - slope_y * axis[:, None]
    pool = torch.nn.functional.max_pool2d
    local_range = pool(height[None, None], 3, 1, 1) + pool(-height[None, None], 3, 1, 1)
    blocked = (~finite) | (level - level[finite].median() > 0.18) | (local_range[0, 0] > 0.08)
    # Include the wheels and turning footprint; cells outside the scan are unknown.
    hard = pool(torch.nn.functional.pad(blocked[None, None].float(), (14, 14, 14, 14), value=1), 29, 1)[0, 0]
    soft = pool(torch.nn.functional.pad(blocked[None, None].float(), (20, 20, 20, 20), value=1), 41, 1)[0, 0]
    # shortcut: height discontinuities approximate rocks, validate a classifier before hazard guarantees.
    return hard * 100 + soft * 0.05


def scan_waypoint(costmap, goal):
    # shortcut: the local grid can stall near tight goals, add calibrated precision tracking before tight-clearance use.
    grid = costmap[::2, ::2].tolist()
    start = (25, 25)
    if grid[25][25] >= 100:
        return None
    distance = {start: 0.0}
    parent = {}
    queue = [(0.0, start)]
    while queue:
        cost, point = heapq.heappop(queue)
        if cost != distance[point]:
            continue
        row, col = point
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            nr, nc = row + dr, col + dc
            if not (0 <= nr < 51 and 0 <= nc < 51) or grid[nr][nc] >= 100:
                continue
            if dr and dc and (grid[row][nc] >= 100 or grid[nr][col] >= 100):
                continue
            candidate = cost + 0.1 * math.hypot(dr, dc) * (1 + grid[nr][nc])
            if candidate < distance.get((nr, nc), math.inf):
                distance[nr, nc] = candidate
                parent[nr, nc] = point
                heapq.heappush(queue, (candidate, (nr, nc)))
    gx, gy = goal.tolist()
    end = min(distance, key=lambda point: math.hypot((point[1] - 25) * 0.1 - gx,
                                                    (point[0] - 25) * 0.1 - gy) + 0.05 * distance[point])
    path = [end]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    traveled = 0.0
    waypoint = start
    for previous, point in zip(path, path[1:]):
        traveled += 0.1 * math.dist(previous, point)
        waypoint = point
        if traveled >= 1.2:
            break
    return goal.new_tensor([(waypoint[1] - 25) * 0.1, (waypoint[0] - 25) * 0.1])


@torch.inference_mode()
def plan_action(model, state, horizon=20, align_heading=False, height_scan=None, terrain_cost=None, hold_reached_goal=True):
    if state.shape != (1, 5) or not torch.isfinite(state).all() or horizon < 1:
        raise ValueError("Planning needs one finite five-field state and a positive horizon")
    if terrain_cost is not None and height_scan is None:
        raise ValueError("Forecast scoring requires measured-scan obstacle checks")
    actions = torch.cartesian_prod(
        state.new_tensor([0.0, 0.05, 0.15, 0.35, 0.65, -0.05, -0.2]),
        state.new_tensor([0.0, -0.35, -0.23, -0.12, -0.04, 0.04, 0.12, 0.23, 0.35]),
    )
    align_heading = align_heading or float(state[0, 2]) / 0.11 < 0.16
    if align_heading:
        actions = actions[actions[:, 0] == 0]
        horizon = min(horizon, 5)
    risk_map = scan_costmap(height_scan) if height_scan is not None else None
    if hold_reached_goal and float(state[0, 2]) / 0.11 < 0.18 and abs(float(state[0, 4])) * math.pi < 0.1:
        return state.new_zeros((1, 2))
    waypoint = None
    if risk_map is not None and not align_heading:
        angle = state[0, 3] * torch.pi
        goal = state[0, 2] / 0.11 * torch.stack((angle.cos(), angle.sin()))
        waypoint = scan_waypoint(risk_map, goal)
        if waypoint is None:
            return state.new_zeros((1, 2))
    lengths = torch.full((len(actions),), horizon, device=state.device, dtype=torch.long)
    turn_steps = torch.zeros_like(lengths)
    follow = actions.clone()
    if risk_map is not None and not align_heading:
        turns = torch.cartesian_prod(state.new_tensor([0.35, 0.65]),
                                     state.new_tensor([-0.35, 0.35]),
                                     state.new_tensor([10., 20., 40., 60.]))
        actions = torch.cat((actions, torch.stack((torch.zeros_like(turns[:, 1]), turns[:, 1]), dim=1)))
        follow = torch.cat((follow, torch.stack((turns[:, 0], torch.zeros_like(turns[:, 0])), dim=1)))
        turn_steps = torch.cat((turn_steps, turns[:, 2].long()))
        lengths = torch.cat((lengths, turns[:, 2].long() + horizon))
    prediction = state.expand(len(actions), -1).clone()
    initial = prediction.clone()
    risk = state.new_zeros(len(actions))
    for step in range(int(lengths.max())):
        command = torch.where((step < turn_steps)[:, None], actions, follow)
        prediction = torch.where((step < lengths)[:, None], predict(model, prediction, command), prediction)
        if risk_map is not None:
            xy = relative_motion(initial, prediction)[:, :2] / 0.11
            index = ((xy + 2.5) / 0.05).round().long()
            outside = ((index < 0) | (index > 100)).any(dim=1)
            step_risk = risk_map[index[:, 1].clamp(0, 100), index[:, 0].clamp(0, 100)]
            risk = torch.maximum(risk, torch.where(outside, 100.5, step_risk))
    distance = prediction[:, 2] / 0.11
    bearing = prediction[:, 3] * torch.pi
    heading = prediction[:, 4] * torch.pi
    if align_heading:
        cost = heading.abs()
    else:
        cost = (relative_motion(initial, prediction)[:, :2] / 0.11 - waypoint).norm(dim=1) if waypoint is not None else distance + distance.clamp(max=2) * (bearing / torch.pi).square()
    if not torch.isfinite(cost).all():
        raise RuntimeError("The learned model produced non-finite planning costs")
    safe = risk < 100
    if terrain_cost is not None and not align_heading and safe.any():
        ticks = torch.arange(25, device=state.device)[None, :, None]
        commands = torch.where(ticks < turn_steps[safe, None, None], actions[safe, None], follow[safe, None])
        predicted_cost = terrain_cost(commands, lengths[safe])
        if (predicted_cost.shape != lengths[safe].shape or not torch.isfinite(predicted_cost).all()
                or (predicted_cost < 0).any() or (predicted_cost > 1).any()):
            raise RuntimeError("Invalid predicted terrain costs")
        cost[safe] += 0.25 * predicted_cost
    cost += risk
    cost[risk >= 100] = torch.inf
    if not torch.isfinite(cost).any():
        return torch.zeros((1, 2), device=state.device, dtype=state.dtype)
    return actions[cost.argmin()].reshape(1, 2)
