# Frozen JEPA versus transport navigation test

This test asks whether the existing v12 JEPA terrain forecasts improve native
goal completion over their matching motion-only terrain transport. It runs
both frozen controllers on twenty new rock layouts, seeds 601–620, with one
trial seed per layout, 100–119. The layout is the independent experimental unit.
The generator, rock count, rock sizes, motion prior, terrain readout, scan
rejection, candidate search, forecast weight and horizon stay fixed. Both
controllers use the tested goal-arrival hold. There is no retraining.

The earlier fresh evaluation reached 6/6 goals with both controllers. Its
roughly 0.7% path reduction does not establish a navigation improvement.

## Fixed decision

Finish all forty trials before testing the claim. Never select or replace a
case after inspecting its outcome. The primary outcome requires native success,
distance below 0.18 m, heading error below 0.1 rad, and no collision. To support
a success improvement, JEPA must meet all four conditions:

- Success rate gain of at least ten percentage points across the twenty pairs.
- Two-sided exact paired discordance test at p <= 0.05.
- No case in which JEPA collides while transport does not.
- Mean JEPA planning time at most 0.2 seconds per command.

The exact test sums the fair-binomial tails of the discordant pairs. Six
JEPA-only successes and zero transport-only successes give p = 0.03125.
Two extra successes alone cannot pass. This small study may not detect a small
benefit. A failure to pass means benefit is not established; it does not prove
equivalence. Equal success cannot be relabeled as success improvement based on
a shorter path. Path length, stalls, latency and command changes are descriptive.

## Execution and checks

`deploy.py` reuses the existing goal-hold job resource configuration. It checks
the committed plan against every source hash, creates an immutable ConfigMap,
and launches one GPU job so every pair uses the same device and dependencies.
The existing pinned runtime installs the same simulator and encoder sources.
Controller order alternates by trial-seed parity. Each pair must reproduce its
initial pose, goal, yaw, height scan, root velocity, joint positions and joint
velocities. Raw terminal-safe traces are checked
independently. No job deadline, timed deletion, or elapsed-time stopping is used.

For every step where forecasts score candidates, the runner also computes the
other controller's action using the same observation and command history.
It records that action without executing it. This audit shows whether learned
corrections change choices. Audit work is excluded from the measured production
planning latency, and real forecast counters are restored afterward.

```bash
rtk proxy uv run --no-project --with pyyaml python deploy/nautilus-jepa-confirmation/deploy.py /path/to/original-navigation-model.pt
rtk kubectl --context nautilus -n razvanlab logs -f job/mars-rover-jepa-confirmation-20261122-v1 -c rover
```

Results and source snapshots persist at
`/data/runs/rover-jepa-confirmation/<pod>/`. `verification.json` is produced only
after every planned trace passes verification. `SUCCESS` means the experiment
completed; inspect `primary.promotion_passed` for the scientific outcome.

## Limits

The test covers new layouts from one synthetic terrain family. It cannot prove
performance on new friction, sensors, or natural terrain. Forecasts cover at
most five seconds. Long turn-then-drive candidates can extend beyond that
horizon, and terminal heading alignment uses no forecast cost. These settings
remain frozen so the experiment tests the current implementation. Inspected
layouts become development data after this evaluation.
