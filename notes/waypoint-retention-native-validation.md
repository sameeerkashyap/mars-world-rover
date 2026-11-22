# Native waypoint retention validation

The predeclared development gate failed. Corrected `649eb0b` reached strict native
success on 9/10 known cases, compared with 5/10 for original `815f19e`, with no
collisions in all 20 trials. Corrected 501/30 lost an original success: native
acknowledgement arrived at 0.164792 m and heading error 0.101741 rad, above the
unchanged 0.1 rad limit. This candidate is not approved for fresh evaluation.

The [compact receipt](waypoint-retention-native-validation.json) preserves every
pair, CSV hash, source and runtime receipt, initial-state checks and gate result.
All 20 trials passed independent CSV and paired-input verification. The failed
conditions are preservation of all original strict successes and success on all
three 501 model-hold regressions. The corrected mean planning time is 0.054842 s,
below the fixed 0.20 s limit. It is the unweighted mean of the ten case means.

## Native outcomes

Pass requires both native acknowledgement and strict terminal geometry. Distances
and heading values below are for the corrected controller; the heading column is
absolute error. All original and corrected trials have zero collisions.

| Layout/seed | Control | Original commands/outcome | Corrected commands/outcome | Final distance m | Heading error rad |
| --- | --- | --- | --- | ---: | ---: |
| 604/103 | transport | 745 timeout | 408 pass | 0.149562 | 0.054963 |
| 611/110 | transport | 745 timeout | 421 pass | 0.144153 | 0.064862 |
| 614/113 | transport | 745 timeout | 337 pass | 0.136597 | 0.081177 |
| 615/114 | transport | 745 timeout | 313 pass | 0.138156 | 0.076364 |
| 601/100 | transport | 304 pass | 423 pass | 0.155146 | 0.061072 |
| 613/112 | jepa | 309 boundary fail | 380 pass | 0.161104 | 0.050470 |
| 601/100 | jepa | 318 pass | 445 pass | 0.152152 | 0.059735 |
| 501/30 | model-hold | 225 pass | 305 boundary fail | 0.164792 | 0.101741 |
| 501/31 | model-hold | 183 pass | 289 pass | 0.169486 | 0.000588 |
| 501/32 | model-hold | 203 pass | 334 pass | 0.164614 | 0.086024 |

All four transport timeouts became strict native successes. Original 613/112
received native acknowledgement outside the distance limit; corrected 613/112
passed. Both 601 controls remained successful but used more commands. Across the
ten pairs there are five corrected-only successes and one original-only success.

The [two-row 501/30 capture](waypoint-retention-501-30-boundary.csv) shows the failed
transition. After command 304, distance is 0.172279 m and heading error is
0.097599 rad. Command 305 repeats speed -0.05 and turn +0.04; native acknowledgement
arrives with heading error 0.101741 rad. The strict result remains a failure.
Heading behavior during arrival needs diagnosis before the development gate is
repeated. No source or threshold was changed in response to these outcomes.

## Fixed comparison

The replay compares original planner `815f19e` with corrected planner `649eb0b`
under one common runner. The correction retains a waypoint only while the current
measured scan still permits it. Release stays fixed at 0.6 m; retention is disabled
within 1.2 m of the true goal and during precision control. No graph, mask,
candidate, horizon, score, forecast weight, learned weight, or native success
threshold changed after the candidate was frozen.

The registered set has ten pairs: transport on 604/103, 611/110, 614/113,
615/114 and 601/100; JEPA on 613/112 and 601/100; and model-hold on 501 seeds
30, 31 and 32. These are known development cases. The transport timeout wins
measure the controller correction with transport scoring. They do not measure a
JEPA benefit or support a claim about new layouts.

## Verification and raw evidence

The existing comparison validator checks source and motion-checkpoint hashes,
runtime receipts, registered terrain mesh arrays, execution order, and full paired
initial state. Initial checks include position, goal, yaw, heading error, root
velocity, joint position, joint velocity and measured scan. The independent CSV
verifier recomputes path, collision, timing and terminal geometry, and requires the
native success flag together with distance below 0.18 m and absolute heading error
below 0.1 rad.

Host verification passed for transport's ten trials, JEPA's four trials and
model-hold's six trials. All ten pairs have zero observed initial-state difference
in each checked field and finite scan sample. Mesh arrays equal the registered
arrays. All modes use the same runtime receipt: A6000, Isaac Sim 6.0.1.0 and pinned
Torch 2.10.0+cu128. The runtime receipt's known pip dependency conflicts remain
recorded with `pip_check_exit_code=1`; pinned runtime checks report `pass`.

Raw CSV, trial JSON, decision HDF5, scans, copied source, terrain and runtime
receipts remain under:

`/data/runs/rover-controller-session/mars-rover-controller-session-a6000-20261122-v1-9gnkn/waypoint-development-649eb0b-v1/`

The immutable source plans are adjacent under
`snapshots/validation-dev-649eb0b14d09-{transport,jepa,model-hold}`.

The local verification export omits decision HDF5 and terrain USD files. It keeps
all CSV, JSON, source, initial/last scans and terrain mesh NPZ files required by
the existing validator. Its archive SHA256 is
`c4a43a539842b087fcb374d7696742596e01d8fe51c2e9596852d6b65930421a`.
The complete raw run remains on the PVC. The full archive hash reported from the
PVC is `e7b1a626e74aa87c46cb2dc4dac19777dc1c39d0c85a4de6538fa272b4913cc0`.
The prelaunch gate hash is
`15ce0a459ae06d942f07e45a18f7e6cc5d22e4858bce3bce7e90c8f7f4609688`;
the copied gate and launch receipt match it. Batch exit zero and `SUCCESS` mean
the native replay completed; `gate_passed` remains false.

To reproduce verification on an exported mode, use its frozen validator and
snapshot. This example checks transport; repeat it for `jepa` and `model-hold`:

```bash
rtk proxy uv run --no-project --python 3.12 --with numpy==2.3.1 python /path/to/snapshots/validation-dev-649eb0b14d09-transport/validation.py verify /path/to/snapshots/validation-dev-649eb0b14d09-transport /path/to/export/transport
```

The formal 4eedd3c arrival comparison remains separate under the same durable
session's `matched-arrival-4eedd3c-v1` output. Its earlier outcomes do not substitute
for any of the registered 649eb0b pairs.
