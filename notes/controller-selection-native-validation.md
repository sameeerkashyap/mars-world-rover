# Combined controller development selection

Exact source `aed6e7030de0c0cd17cd9fa0a303e7ec21e6cbe8` passes the unchanged
development gate. The combined waypoint and arrival controller reaches strict
native success on 10/10 registered known cases, versus 5/10 for original
`815f19e`. There are five corrected-only successes, no lost original successes,
and zero collisions across all 20 trials. Mean corrected planning time is
0.055407 s, below the fixed 0.20 s bound. This is the unweighted mean of the ten
case means.

The [compact receipt](controller-selection-native-validation.json) contains all
ten pairs, CSV hashes, the prelaunch gate, measured gate values and source/runtime
receipts. Every required condition passes. This exact source is selected for the
registered fresh comparison. Development cases do not establish performance on
new layouts, a statistical endpoint, or a JEPA benefit.

## Outcomes

Pass requires native acknowledgement, no collision, terminal distance below
0.18 m and absolute heading error below 0.1 rad. Final geometry below is for the
corrected controller. All original and corrected trials have zero collisions.

| Layout/seed | Control | Original commands/outcome | Corrected commands/outcome | Final distance m | Absolute heading error rad |
| --- | --- | --- | --- | ---: | ---: |
| 604/103 | transport | 745 timeout | 408 pass | 0.149562 | 0.054963 |
| 611/110 | transport | 745 timeout | 421 pass | 0.144153 | 0.064862 |
| 614/113 | transport | 745 timeout | 337 pass | 0.136597 | 0.081177 |
| 615/114 | transport | 745 timeout | 313 pass | 0.138156 | 0.076364 |
| 601/100 | transport | 304 pass | 423 pass | 0.155146 | 0.061072 |
| 613/112 | jepa | 309 boundary fail | 380 pass | 0.161104 | 0.050470 |
| 601/100 | jepa | 318 pass | 445 pass | 0.152152 | 0.059735 |
| 501/30 | model-hold | 225 pass | 305 pass | 0.168569 | 0.096337 |
| 501/31 | model-hold | 183 pass | 289 pass | 0.169486 | 0.000588 |
| 501/32 | model-hold | 203 pass | 334 pass | 0.164614 | 0.086024 |

All four transport timeouts become strict native successes. These wins measure
the controller correction with transport scoring. Both 601 controls retain their
successes but take more commands: 304 to 423 for transport and 318 to 445 for
JEPA. This correction does not improve travel time in every known case.

The selected source combines currently reachable waypoint retention with the
arrival margin brake. The brake uses the existing zero-command hold when heading
has less normalized boundary margin than distance. Its full 501/30 regression now
passes at 0.168569153 m and 0.096337080 rad. The earlier
[649eb0b failed gate](waypoint-retention-native-validation.md) and rejected
[heading candidate evidence](arrival-heading-correction.md) remain preserved.
The complete current comparison establishes selection for `aed6e70`.

## Verification and reproduction

All three frozen validators were independently rerun on the host: transport has
ten trials, JEPA four and model-hold six. The validators recompute outcomes from
CSV and check registered cases, execution order, source/checkpoint hashes,
runtime receipts, terrain mesh arrays and full paired initial state. Maximum
paired difference is zero for position, goal, yaw, heading error, root velocity,
joint position, joint velocity and finite scan samples. All mesh arrays equal the
registered arrays. The source and runner stay fixed throughout the replay.

The prelaunch gate's required conditions exactly equal the earlier gate's rules.
Its SHA256 is `521d2c53c2ba922c7251680051d7c98d0c2b75857fc5ea37470ebc6cde637fb7`.
The copied gate, launch receipt, launcher and all three source-check receipts
match. Source `2dffed4` adds a measured transition fixture and test; runtime source
is unchanged from selected `aed6e70`. Selection uses the exact frozen `aed6e70`
snapshot, not a moving branch tip.

Durable raw output is:

`/data/runs/rover-controller-session/mars-rover-controller-session-a6000-20261122-v1-9gnkn/margin-brake-development-aed6e70-v1/`

Immutable snapshots are adjacent under
`snapshots/validation-dev-aed6e7030de0-{transport,jepa,model-hold}`.
The filtered export archive SHA256 is
`7bea772223fa6e1fc3a82f75397d5ab02d3f1c9952f1e7e327e3c065f88bf5a8`.
The full 340-file raw manifest was computed read-only on Nautilus; its SHA256 is
`9ef631f513af975a3d42c1aff33a9b19cc1bb1e0f98b1931095dc45897f6f96a`.
Every exported raw file independently matches its manifest hash on the host.
Decision HDF5, terrain USD and outer `console.log` stay on the PVC and were not
transferred in this export. The receipt preserves runtime versions and the known
pip dependency-check result separately from the passing pinned-runtime checks.

Use the frozen validator to reproduce checks on an exported mode. Repeat this
command for `jepa` and `model-hold` with their matching snapshots and output paths:

```bash
rtk proxy uv run --no-project --python 3.12 --with numpy==2.3.1 python /path/to/snapshots/validation-dev-aed6e7030de0-transport/validation.py verify /path/to/snapshots/validation-dev-aed6e7030de0-transport /path/to/export/transport
```
