# Shared timeout diagnosis

Ticket [17](https://github.com/sameeerkashyap/mars-world-rover/issues/17) remains open for a native replay. The eight saved trials show repeated turn reversals and little motion. Their last measured scans permit local routes and safe candidates. The saved data does not establish one shared cause.

The raw input is `/private/tmp/rover-wayfinder/confirmation-raw`. Its durable root is `/data/runs/rover-jepa-confirmation/mars-rover-jepa-confirmation-20261122-v1-rg5hp/`. Each case has its full CSV, trial JSON and initial/last scan archive. [The saved numerical report](rover-shared-timeouts-analysis.json) includes all input hashes. It was produced before the arrival merge, with frozen planner SHA256 `0c12c0c42059a972ad7b67bff16f2831369291e90d2cc456fa19cb8e44d73415` and motion checkpoint SHA256 `76953ac81e383a6000d9b26bf8026b5049f4ce768831bd48506dea31d7649283`.

## Reconstruction and fast failure signal

The runner saves `last` before the last command. The CSV records poses after commands. All eight trials end at CSV step 745. Their saved last scans therefore match the pose at step 744 and the previous command in row 744. The command selected from that scan is in row 745. Using the terminal pose with this scan would mix times.

The reconstructed policy state is `[previous_speed, previous_turn, planar_distance*0.11, wrapped_goal_bearing/pi, heading_error/pi]`. The upstream target command uses the rover's yaw rotation, rather than its full tilted pose. The trace's fixed goal and recorded yaw therefore give the measured planar state. This reconstructs the expected state; it cannot prove that the original live policy fields were current.

Run from the repository root with the cached environment:

```bash
rtk proxy uv run --no-project --python 3.12 --with torch==2.10.0 --with torchvision==0.25.0 --with numpy==2.3.1 --with h5py==3.15.1 --with einops==0.8.1 --with pyyaml --with gdown==6.5.1 python -m scripts.diagnose_rover_timeouts /private/tmp/rover-wayfinder/confirmation-raw/fresh-test-rocks-{604,611,614,615}/*/seed-*.csv --checkpoint /Users/sameerkashyap/code/mars_world_rover/runs/rover-baseline/20261113T210435968626Z/model.pt --output /private/tmp/rover-controller/timeout-telemetry-parity.json --require-success
```

The initial throwaway reconstruction ran in 1.6 seconds and failed with `Native timeouts persist in captured trials: [604, 611, 614, 615]`. The committed command also checks optional-diagnostics action equality and standard-candidate scoring against public `plan_action`. It ran in 4.0 seconds for all eight cases, wrote the report, then exited nonzero with `Native timeout or failure remains in saved trace`. This is a saved-trace symptom check. Corrected native traces are required to turn it green; offline planning cannot change the recorded outcomes. Without `--require-success`, the same command produces a diagnosis report successfully.

## Ranked hypotheses and probes

These four hypotheses were sent before the probes:

1. Replanning restarts turn-then-forward plans before their forward phase. Prediction: long turn-only runs, frequent reversals, and a safe candidate whose forward phase is never reached. The trace and final candidate selection support this pattern. They cannot show which candidate or waypoint was selected earlier.
2. The sparse candidate set misses a usable motion. Prediction: reachable measured routes, with a large improvement when turn durations become dense. Replacing only `[10,20,40,60]` by `1..60` increases the set from 79 to 303 candidates. Costs improve, but every case still selects the same first command `[0,-0.35]`. A denser duration set alone does not remove the final turn decision.
3. The footprint mask disconnects the route. Prediction: no reachable local waypoint or no safe motion candidates. This is rejected for the saved final scans. All starts and selected local waypoints are reachable under the unchanged hard mask, and all cases have safe forward-first candidates. Complete goal reachability remains unknown because each goal is outside the scan.
4. Off-scan goals and local route changes make the controller cycle. Prediction: goals outside the scan, repeated nearby poses, and route/turn changes under small heading changes. The first two are measured. A fixed-world-pose scan rotation probe can switch waypoint side and command sign on 604 and 615. It does not do so on 611 or 614 over yaw offsets from -0.1 to +0.1 rad. This does not establish a common route-switching cause.

The yaw probe rotates both the saved scan and goal consistently. It uses nearest-cell height samples and marks rotated unknown corners as NaN. For 604 transport, yaw offset -0.05 rad changes the waypoint from world-relative `[0.5,-1.1]` to approximately `[0.544,0.874]` and selects `[0.15,+0.35]`. For 615 transport, +0.025 rad switches the waypoint side and turn sign. Scan interpolation is an offline sensitivity probe, not a physical replay. Changing only the previous turn field to -0.35, 0, or +0.35 leaves all eight final decisions at `[0,-0.35]`.

## Per-case evidence

J means JEPA; T means transport. All eight native outcomes are time-limit failures with no collision. All final scan-only commands are `[0,-0.35]`. The hard footprint half width stays 0.7 m.

| Layout / seed / control | Final distance m | Reachable local cells | Distance to hard mask m | Safe / 79 | Safe forward-first | Turn-only / last 100 | Turn sign changes / last 100 | Net motion / last 100 m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 604 / 103 / J | 3.236 | 467 | 0.35 | 48 | 18 | 48 | 86 | 0.028 |
| 604 / 103 / T | 3.227 | 435 | 0.25 | 48 | 18 | 47 | 87 | 0.040 |
| 611 / 110 / J | 8.115 | 1218 | 1.15 | 68 | 27 | 100 | 93 | 0.070 |
| 611 / 110 / T | 8.036 | 1199 | 1.05 | 67 | 27 | 100 | 91 | 0.003 |
| 614 / 113 / J | 8.762 | 966 | 0.30 | 44 | 9 | 100 | 94 | 0.045 |
| 614 / 113 / T | 8.762 | 966 | 0.30 | 44 | 9 | 100 | 94 | 0.045 |
| 615 / 114 / J | 3.977 | 1078 | 1.19 | 68 | 27 | 100 | 79 | 0.067 |
| 615 / 114 / T | 4.020 | 1096 | 1.18 | 68 | 27 | 100 | 81 | 0.065 |

Distance to the hard mask is the nearest blocked cell center after footprint inflation. It is a local grid margin, not measured body-to-rock clearance. The saved report calls this field `remaining_center_clearance_m`; the command now calls it `distance_to_hard_mask_m`.

The standard winners plan a 40-step turn followed by speed 0.35 on 604 and 614, a 20-step turn followed by speed 0.35 on 611, and a 20-step turn followed by speed 0.65 on 615. These steps use the frozen learned motion model. They are predictions, not observed executed plans. The longest observed turn-only runs are 31 steps on 604, 716 on 611, 745 on 614, and 632/635 on 615 J/T. Final-100-step yaw ranges remain between 0.127 and 0.168 rad.

## Native decision capture and next discriminating replay

The optional `plan_action(..., diagnostics=dict)` output records the selected candidate index, turn duration, follow command, rollout length, safe counts, selected measured risk, total cost and top-two cost gap. It also records the local waypoint and explicit reasons for hold, blocked start, no safe candidate, or omitted forecast scoring. This output does not change candidate generation, scores, weights, safety rejection or commands.

Run `rover_navigate.py` with `--record-decisions` to create `seed-<seed>-<controller>-decisions.h5`. The file contains exact pre-command `policy` values, `position_w`, `yaw_rad`, the selected `command`, and JSON `diagnostics`, including `world_waypoint_m`. Only rows before `steps_written` are valid. `complete` means the command loop ended normally. The CSV and JSON still establish native outcomes.

```text
HDF row k: policy + scan + pose -> chosen command -> CSV step k+1: resulting pose
```

The fixed goal comes from the trial JSON's `initial.goal`. The fixed target heading is `initial.yaw + initial.heading_error_rad`, modulo 2*pi. The runner checks that the goal does not change. This allows a precise comparison of live policy distance/bearing/heading against measured pre-command pose and goal. Reject the stale-policy hypothesis if they agree. If they disagree, offline measured-state planning must remove the lag before any control change is justified.

The next native replay uses the same four known layouts and seeds, the same two frozen controls, and the existing 750-control-step episode limit. Record the baseline first. Inspect consecutive world waypoints, candidate turn durations, commands, and live/measured goal states across each reversal. Distinguish route-side switching, restarted turn plans and state timing before choosing a correction. If needed, test one change at a time: retain a world waypoint, or retain progress through a selected turn plan. Recheck every proposed step against the current measured scan. Do not issue an unchecked multi-step command. No controller correction is included here.

Known layouts can now run under an explicitly frozen development forecast plan. Case membership, seed membership, source hashes, weights and plan step counts remain validated. The legacy two-controller arrival replay and its evidence are unchanged.

## Checks and limits

TDD checks covered the CLI reconstruction, its native-success failure signal, exact native-runner pre-command recording, optional-recording action parity, and planner diagnostics with a forecast callback. The focused planner, forecast, confirmation, navigation-exit and new diagnosis suites pass: 24 tests in 5.7 seconds. The eight real saved scans also pass diagnostics-on/off action equality. A missing worktree `external` link caused two initial environment failures; adding the existing upstream source link resolved them. The link is not part of the commit.

No training, weight change, safety/native threshold change, Nautilus launch or job termination was performed. No physics claim follows from these tests. Final-scan-only data cannot prove earlier scan coverage, historical waypoint/candidate choices, full goal reachability, physical clearance along a future turn, historical forecast costs, or a shared causal mechanism. The native capture replay is still required.
