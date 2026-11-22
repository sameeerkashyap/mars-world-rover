# Native shared-timeout diagnosis

All four known timeout layouts show large world-waypoint switches with repeated command reversals and little motion. Live goal-state timing is current, and safe forward-first candidates remain available. This supports a stateful waypoint hypothesis for a controlled correction replay. It does not establish that a particular correction works.

The [compact aggregate](rover-native-timeouts-analysis.json) covers the eight timeout arms and the two successful layout 613 arms. It records input hashes, frozen source hashes, analyzer/helper hashes, timing checks, candidate summaries and prior-waypoint connectivity counts. Per-step and per-switch details remain in the raw decision HDF5 files. No controller code was changed for this analysis.

## Inputs and timing

These are known development cases, not fresh confirmation evidence. The completed native batch used immutable source commit `68ff236449953775eacc1b3e178f5f167df589fd`. The durable batch is `/data/runs/rover-controller-session/mars-rover-controller-session-a6000-20261122-v1-9gnkn/development-68ff236-v2`. Raw exports contain matching CSV, trial JSON, decision HDF5, initial/last scan archive, config and copied source files. [The arrival receipt](rover-arrival-native-validation.json) includes the independent native CSV verification and source/weight checks for the whole batch.

For every analyzed trial, the capture is complete and its row count matches the CSV and trial JSON. Command rows match CSV commands exactly at stored float32 precision. HDF pose row 0 matches the settled initial pose; later pose rows match the preceding CSV sample. Policy previous-command fields match the preceding command. Initial and final saved scans match the corresponding HDF observations. Recorded world waypoints also match the local waypoint transformed through the contemporaneous pose.

The fixed target heading is initial yaw plus initial heading error, wrapped into [-pi,pi). Every live policy distance, bearing and heading agrees with the measured pre-command pose and fixed target to numerical precision. Across all ten traces, maximum errors are below 4e-6 m for distance and 7e-7 rad for bearing and heading. This rejects stale live goal-state timing for these cases. It does not certify a height scan's physical timestamp or registration against terrain.

## Shared observed pattern

J means JEPA; T means transport. A large jump is a change above 1 m between consecutive recorded world waypoints. This is an analysis definition, not a control threshold.

| Layout/control | Outcome | Large jumps, full trace | Immediate turn reversals | Large jumps coincident with reversal | Last100 median jump m | Last100 turn-only steps | Minimum safe forward-first candidates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 604 J | timeout | 304 | 320 | 304 | 1.968 | 48 | 18 |
| 604 T | timeout | 304 | 316 | 304 | 2.058 | 47 | 18 |
| 611 J | timeout | 666 | 666 | 665 | 1.731 | 100 | 27 |
| 611 T | timeout | 692 | 692 | 691 | 1.794 | 100 | 27 |
| 614 J | timeout | 674 | 674 | 674 | 2.278 | 100 | 9 |
| 614 T | timeout | 674 | 674 | 674 | 2.278 | 100 | 9 |
| 615 J | timeout | 597 | 612 | 597 | 1.741 | 100 | 27 |
| 615 T | timeout | 601 | 611 | 601 | 1.744 | 100 | 27 |
| 613 J | native success | 0 | 25 | 0 | 0.006 | 0 | 36 |
| 613 T | native success | 0 | 207 | 0 | 0.086 | 7 | 0 |

The last 100 pre-command poses move only 0.003–0.070 m in the timeout cases. Successful 613's largest waypoint changes are 0.138 m J and 0.176 m T. This contrast is stronger than reversal count alone: successful 613 transport also has 207 reversals. Safe forward-first candidates are never absent in the timeout cases. Their stalls are not universal hard rejection of forward motion.

All 745 decisions on 614 select a 40-step turn prefix followed by forward motion. The executed controller never completes a 40-command identical turn prefix there and never issues the selected follow command at the original prefix horizon. The other timeout cases also replan before nearly every selected identical turn prefix completes. Successful 613 does so as well, so prefix interruption alone does not establish the failure mechanism. A replanned controller is not required to execute an old candidate schedule.

Candidate cost gaps during reversals are often substantial. The aggregate keeps their minimum, median and maximum. These are gaps between candidate plans; they need not measure the best alternative first command. The data does not show a shared numerical tie-breaking failure.

## Current-scan prior-waypoint checks

The falsifiable prediction was declared before this probe: unnecessary switching should leave the previous world waypoint in the current measured start component. For every large jump in the last 100 decisions, the old world point was transformed into the current body frame. Its hard-mask cell and graph reachability were checked separately. The 0.1 m graph uses existing `measured_route`; a diagnostic 0.05 m graph uses the same hard mask and diagonal corner rule. Neither footprint nor obstacle thresholds changed.

| Layout/control | Tested large switches | Old point hard-free | Old point reachable on0.1m graph | Old point reachable on0.05m graph |
| --- | ---: | ---: | ---: | ---: |
| 604 J | 87 | 87 | 43 | 70 |
| 604 T | 88 | 88 | 44 | 74 |
| 611 J | 93 | 93 | 93 | 93 |
| 611 T | 92 | 92 | 92 | 92 |
| 614 J | 95 | 95 | 95 | 95 |
| 614 T | 95 | 95 | 95 | 95 |
| 615 J | 80 | 80 | 80 | 80 |
| 615 T | 81 | 81 | 81 | 81 |

For 611, 614 and 615, every tested prior waypoint remains both free and reachable when the planner switches world-side. Retaining a currently reachable world waypoint is therefore a supported correction hypothesis for a controlled native test.

For 604, none of the unreachable coarse target cells is itself blocked. Those free cells lie outside the current start component. The original detailed probe also found no reachable coarse node within 0.1 m of those old targets. Some targets become connected on the finer graph, showing a sampling effect, but 17 J and 14 T remain disconnected even at 0.05 m. Free target cells alone do not establish reachable paths. A retained waypoint must be discarded or replaced when current measured reachability fails. These observations do not justify weakening a mask, changing the production graph resolution or issuing an unchecked multi-step turn.

## Reproduction and checks

From the repository root, replace `/path/to/native-dev` with the exported batch root:

```bash
rtk proxy uv run --no-project --python 3.12 --with torch==2.10.0 --with torchvision==0.25.0 --with numpy==2.3.1 --with h5py==3.15.1 --with einops==0.8.1 --with pyyaml --with gdown==6.5.1 python -m scripts.analyze_rover_decisions /path/to/native-dev/test-rocks-{604,611,613,614,615}/*/seed-*.csv --output native-timeout-analysis.json
```

The CLI accepts the same positional CSV and optional `--output` convention as the other diagnosis tools. It verifies source receipts, completed capture shape, command/pose/scan alignment and world-waypoint transforms. A focused functional check uses the actual runner's recorded output, detects a changed live distance and rejects a changed recorded command. CLI assertions independently check a free point separated by a wall, then a one-cell opening reachable on the fine graph but absent from the coarse graph. The complete ten-trial CLI run reproduces the earlier external analysis and retention counts. The focused diagnosis and planner suites pass all 11 tests.

No training, weight change, safety/native threshold change, job launch, source ConfigMap mutation or job termination was performed. The separate global geometry certificates show complete planar routes, but privileged geometry is never a controller input. A bounded native replay of a specific correction is still required before claiming a causal fix or improved reliability.
