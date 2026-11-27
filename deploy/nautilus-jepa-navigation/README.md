# Frozen world-model navigation comparison

This is a completed historical comparison. Its original source is preserved
under the shared run directories. The current planner holds at a reached goal;
its source hashes differ from these frozen plans. Reproduce the old runs from
their saved source snapshots. The launcher refuses an existing job and changed
source fails hash checks. Inspected layouts are now development data.

See [consolidated findings](../../docs/results.md)
and the separate [goal-hold replay](../nautilus-goal-hold/README.md).

This evaluation compares three controllers with the same goal-motion model,
candidate commands, scan-derived route, and measured-scan obstacle checks:

| Controller | Extra candidate cost |
| :--- | :--- |
| `model` | None; the existing scan-aware planner |
| `transport` | Terrain-cost readout on motion-only transported features |
| `jepa` | The same readout on the frozen v12 DINO-WM rover forecasts |

The v12 predictor passed the earlier quality checks. The later scaled predictor
missed its fixed validation target and is excluded. The readout is the full-data
model selected at epoch 53 of the completed scale run. File hashes, source
hashes, cases, and scoring parameters are fixed in [test-plan.json](test-plan.json).
The launcher creates an immutable ConfigMap. No training takes place here.

## How the planner uses forecasts

```text
Measured scan → route and hard candidate rejection
                                 ↓
Actual observation/action history + safe candidate commands
                                 ↓
                  Predict 1–5 seconds of terrain features
                                 ↓
                  Read terrain costs and rerank safe paths
                                 ↓
                     Execute one bounded command
```

`src/rover_forecast.py` keeps six observations at 0.2-second intervals and the
five actual commands between them. The first and last features form the two
one-second-spaced context frames. Observation previous-command fields must match
the command actually applied. History is cleared between trials and after invalid
scans. The first five control steps use the existing planner while history fills.

Candidates are scored only after they pass the measured-scan checks. The
predictor receives at most 25 future control commands, grouped into one-second
blocks. The readout scores the central four patches in each predicted rover
frame. The maximum active-horizon cost is multiplied by 0.25 and added to the
existing waypoint cost. Horizons beyond five seconds are not forecast. Final
heading alignment keeps the existing scan-only scoring. Unknown or blocked
measured paths cannot be made valid by a learned score.

The learned cost is a height-rule blocked-area fraction. It is not a calibrated
collision probability or a physical safety guarantee.

## Cases and execution

Development cases are debug seeds 4–6 and Mars seeds 7–9. Fresh cases use two
new, deterministic obstacle layouts with layout seeds 501 and 502 and trial
seeds 30–35. Each fresh layout has an 80 m ground mesh around a 30 m obstacle
region and 80 rocks at new positions and sizes. The mesh arrays are saved with the results. These are
unseen obstacle layouts; they do not establish generalization to different
friction or natural terrain classes. Settings are frozen before any fresh
outcome is inspected.

The comparison covers 36 trials: three controllers on six development cases
and six fresh cases. Matched trials must have the same measured initial pose and goal.
Robot joints reset to their defaults with zero velocity. The existing collision,
goal, far-from-goal, and episode termination checks remain enabled.

```bash
rtk proxy bash deploy/nautilus-jepa-navigation/deploy.sh
rtk kubectl --context nautilus -n razvanlab logs -f job/mars-rover-jepa-navigation-20261120-v2 -c setup
rtk kubectl --context nautilus -n razvanlab logs -f job/mars-rover-jepa-navigation-20261120-v2 -c rover
```

The job uses the existing runtime installer and GPU storage. DINOv2 source and
weights are prepared in the setup container for offline inference. No image
rebuild is required. There is no elapsed-time cutoff or automatic timed deletion
in the job manifest. The launcher refuses an existing job instead of replacing
its source.

The v1 job completed all 18 development trials, then failed before any fresh
control step. The original 30 m ground did not fit the upstream sampler's
25 m border on each side. The v2 retry adds ground padding and runs only the
18 fresh trials. Rock vertices are byte-for-byte unchanged. It links the saved
development directories instead of rerunning them. `test-plan-v1.json` preserves
the original plan; the new plan records its hash. Verification permits this
terrain-setup change only when controller source, weights, cases, and scoring
settings match the original plan.

## Evidence

Results persist under `/data/runs/rover-jepa-navigation/<pod>/`. Each phase and
terrain has its own trial directory. CSV traces preserve terminal poses before
automatic reset, commands, collision/success flags, planning latency, and history
readiness. JSON reports also record path length, five-second stop stalls,
forecast-scoring calls, and mean/p95 planning latency. Latency includes live
encoding for the forecast controllers. GPU operations synchronize before the
measurement ends.

`verify.py` independently recomputes distances, heading errors, path lengths,
command bounds, stalls, and success/collision counts from the CSV traces. It
checks all three controllers started from matched states. `verification.json`
compares goals reached, collisions, stalls, path length, and planning latency
separately for development and fresh cases.

Success requires distance below 0.18 m, heading error below 0.1 rad, and no
collision. `SUCCESS` marks a completed and checked evaluation. It does not mean
the learned controller beat either baseline. Read the measured comparison.
