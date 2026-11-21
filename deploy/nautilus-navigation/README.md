# Scan-aware learned-model rover navigation

For the frozen v12 terrain-forecast integration and matched scan-only,
motion-transport, and learned-forecast comparison, see
[world-model navigation](../nautilus-jepa-navigation/README.md).

This evaluation connects the trained `goal-motion-v1` checkpoint to RLRoverLab
in Isaac Sim. It compares three controllers on matching seeds and goals:

- `model`: the learned motion model with local height-scan routing.
- `goal-only`: the original learned-model planner without scan routing.
- `forward`: the scripted forward command.

Robot joints reset to default positions and zero velocity. Five zero-command
steps let the rover settle before each trial. Collision detection, time limits,
and the far-from-goal termination remain enabled.

## How scan routing works

The existing scanner measures a 5 m square at 5 cm spacing. The planner checks
that the scan uses the configured `xy` order and follows rover yaw. It removes
the overall ground slope, marks raised or abrupt height changes, and treats
missing scan cells as unsafe. A 70 cm margin includes the wheels and the turning
footprint. A larger margin adds a small preference for more clearance.

The planner finds a short route through the observed free cells on a 10 cm grid
and follows a point 1.2 m along that route. The learned model predicts the motion
of 63 constant-command candidates and 16 turn-then-drive candidates. Predicted
paths that enter unsafe or unseen cells are rejected. If every candidate is
unsafe, the rover receives a stop command.

Near the goal, nine zero-speed turn candidates align the final heading over five
predicted steps. The planner replans after every actual control step. The model
weights are unchanged from the original goal-only evaluation. The new routing
uses policy height observations, not the simulator's private obstacle map.

The height rules are an approximation. A local route can stop near a tight goal
or a narrow passage. These limits need measured low-speed control and more
terrain tests before making safety claims.

## Run the evaluation

From the repository root, with Nautilus access and the existing shared volume:

```bash
rtk proxy bash deploy/nautilus-navigation/deploy.sh runs/rover-baseline/20261113T210435968626Z/model.pt
rtk kubectl --context nautilus -n razvanlab logs -f job/mars-rover-navigation-v4 -c setup
rtk kubectl --context nautilus -n razvanlab logs -f job/mars-rover-navigation-v4 -c rover
```

The launcher submits debug seeds 4–6 and Mars seeds 7–9 as separate finite Jobs.
It freezes the source and checkpoint in an immutable ConfigMap and refuses to
replace an existing Job. Each manifest requests one RTX A6000. The fresh test
uses seeds 10–12 on debug terrain and 13–15 on Mars terrain, selected before the
test. Its exact manifests are saved with the raw run artifacts.

Native success requires distance below **0.18 m** and heading error below
**0.1 rad**, with no collision. A recorder hook captures terminal poses before
Isaac automatically resets the rover. The evaluator checks the native flag
against the measured terminal pose.

Results persist under `/data/runs/rover-navigation/<pod-name>/`:

- `config.json`: source and checkpoint hashes, seeds, settings, and thresholds.
- `source/`: the Python files used by that evaluation.
- `seed-<seed>-<controller>.csv`: measured poses, commands, and termination flags.
- `seed-<seed>-<controller>.json`: progress, path length, success, and collisions.
- `seed-<seed>-<controller>-scan.npz`: initial and last policy height scans.
- `summary.json`: all trials and controller comparisons.
- `SUCCESS`: written only when a measured goal success occurs and mean scan-aware
  progress exceeds forward progress.

Logs, checkpoint, hardware, and stack information persist under
`/data/runs/rover/<pod-name>/`. Installation versions remain under
`/data/runs/setup/<pod-name>/`. The wrapper checks the evaluator's own success
marker because Isaac shutdown can otherwise mask a Python error with exit code
zero. Read the metrics rather than inferring model quality from Job status.

See [results](../../docs/results.md) for measured outcomes and saved evidence.
RGB capture remains disabled. Trajectory plots show measured positions.
