# Frozen fresh controller comparison v1

This protocol registers 20 new-layout pairs before any fresh outcome is opened.
It implements [ticket 18](https://github.com/sameeerkashyap/mars-world-rover/issues/18)
under [spec 16](https://github.com/sameeerkashyap/mars-world-rover/issues/16).
The selected controller passed the fixed known-case gate; retain its
development reports in external run storage. Those inspected cases stay outside this study.
This commit freezes the plan. Launch must wait until the integration branch has
merged and pushed it, and the root agent has explicitly approved progression.

## Fixed source and snapshot

The original planner is `815f19e8858bd47361a868fb7500e52d2449e76f`.
The corrected planner, common runner, verifier, terrain generator and setup are
from exact commit `aed6e7030de0c0cd17cd9fa0a303e7ec21e6cbe8`.
Both roles use that common runner and differ only in the selected planner.
The [generated plan](fresh-v1-test-plan.json) contains every source and weight
hash. The external snapshot manifest `SHA256SUMS` lists 17 payload files, including
the plan. The manifest itself is the eighteenth snapshot file. Paths in the
manifest are relative to the snapshot, not this folder.

| Receipt | SHA256 |
| --- | --- |
| `test-plan.json` | `8b3715ccb5dcf303ad812922d64aea7a6c907585f2ffb8ed4cdd1772ed73482a` |
| `SHA256SUMS` (snapshot identity) | `6a5342133f4851d50be9ed8c04f9560c928b20f524be84806ca3749239825666` |
| Session `stack.json` | `b4bd94a3f3fe51194f85f09155297fd1af59b470a092bf52adb4aed4152e44bd` |
| Session `hardware.txt` | `de7cf49a6ce4ed78f7fe1390b3ef5693d4cfa35bd94b33bbe692fc79f99ef648` |

The new local snapshot is
`/private/tmp/rover-controller/validation-fresh-aed6e7030de0-transport-v1`.
All 18 files and all snapshot directories have no write bits. It was prepared
with the existing tool:

```bash
rtk proxy uv run --no-project --python 3.12 --with numpy==2.3.1 python deploy/nautilus-controller-validation/validation.py prepare aed6e7030de0c0cd17cd9fa0a303e7ec21e6cbe8 /private/tmp/rover-controller/validation-fresh-aed6e7030de0-transport-v1 --phase fresh --mode transport --checkpoint /Users/sameerkashyap/code/mars_world_rover/runs/rover-baseline/20261113T210435968626Z/model.pt --predictor-source /Users/sameerkashyap/code/mars_world_rover/external/jepa-wms/app/plan_common/models/vit.py
```

Use the ready Nautilus session, context `nautilus`, namespace `razvanlab`, pod
`mars-rover-controller-session-a6000-20261122-v1-9gnkn`, container `rover`.
Its root is
`/data/runs/rover-controller-session/mars-rover-controller-session-a6000-20261122-v1-9gnkn`.
The runtime receipt is `<session>/stack.json`. Hardware is RTX A6000, driver
615.71.10, Isaac Sim 6.0.1.0, Python 3.12.13, Torch 2.10.0+cu128 and NumPy 2.3.1.
Keep the existing stack. Its pinned checks pass; the receipt also records the
existing nonzero `pip check` result. Do not upgrade packages for this study.

Upload the snapshot to the new path
`<session>/snapshots/validation-fresh-aed6e7030de0-transport-v1`.
Check every file against `SHA256SUMS` after upload and save that receipt.
Check the manifest, plan, runtime and hardware hashes above before launch.
Do not overwrite a snapshot or output. No new Job or bootstrap is needed.

All weights stay frozen. The motion checkpoint is copied into the snapshot.
The other inputs stay at these existing session paths:

| Input | Path or exact reference |
| --- | --- |
| Motion checkpoint | Snapshot `model.pt`; SHA256 `76953ac81e383a6000d9b26bf8026b5049f4ce768831bd48506dea31d7649283` |
| JEPA run | `/data/runs/rover-jepa/mars-rover-jepa-train-v12-dfj2t`; checkpoint `training/best.pt` |
| Terrain readout | `/data/runs/rover-jepa-scale/mars-rover-scale-20261120-v1/terrain-cost-mars-rover-scale-20261120-v1-s9ckw/best.pt` |
| DINOv2 source | `/runtime/dinov2`, commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8` |
| DINOv2 weights | `/data/cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` |
| JEPA predictor source | Snapshot `meta_vit.py`, from `jepa-wms` commit `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0` |

The live JEPA, readout and DINOv2 weight hashes were checked before registration
and match the generated plan. No retraining or model selection is part of this
comparison. Transport mode uses the frozen encoder, motion transport and terrain
readout; it does not use the learned JEPA forecast as the comparison factor.

## Cases and execution

Run one original and one corrected trial for each row, in row order. Each pair
uses fixed `transport` mode, the same reset seed, terrain, weights and runtime.
There are 40 trials. Planner order alternates in the existing comparison runner.
The plan's `counterbalance_order: false` refers to controller-mode order within a
trial; there is only one mode. It does not disable this planner alternation.

| Layout | Reset seed | First planner | Second planner |
| --- | ---: | --- | --- |
| 701 | 200 | original | corrected |
| 702 | 201 | corrected | original |
| 703 | 202 | original | corrected |
| 704 | 203 | corrected | original |
| 705 | 204 | original | corrected |
| 706 | 205 | corrected | original |
| 707 | 206 | original | corrected |
| 708 | 207 | corrected | original |
| 709 | 208 | original | corrected |
| 710 | 209 | corrected | original |
| 711 | 210 | original | corrected |
| 712 | 211 | corrected | original |
| 713 | 212 | original | corrected |
| 714 | 213 | corrected | original |
| 715 | 214 | original | corrected |
| 716 | 215 | corrected | original |
| 717 | 216 | original | corrected |
| 718 | 217 | corrected | original |
| 719 | 218 | original | corrected |
| 720 | 219 | corrected | original |

Keep the registered 750-step limit, five settling steps, native termination and
collision rules, strict distance <0.18 m and absolute heading error <0.1 rad.
Keep measured-scan rejection and all planner masks, candidate bounds and forecast
settings unchanged. A timeout or boundary miss remains a failure. A native
success flag alone is insufficient. No inspected layout enters fresh inference.

After progression is approved, run the frozen snapshot's `validation.py run` with
absolute snapshot and new output paths, `--runtime-receipt <session>/stack.json`,
`--jepa-run` and `--terrain-checkpoint` from the table, and
`--encoder-source /runtime/dinov2`. Use output
`<session>/fresh-controller-aed6e70-transport-v1`. Keep launcher, upload and hash
receipts with the output. Preserve full CSV, JSON, scans, meshes, decision HDF5,
source copies and runtime data on the PVC. Do not add an elapsed deadline,
timeout wrapper or timed deletion. Root owns the session's final `DONE` marker.

## Verification and decision

Run the frozen `validation.py verify SNAPSHOT OUTPUT` independently before
analysis. It must verify all 40 trials, the complete execution order, exact
source/input hashes, runtime receipts, generated mesh arrays and paired initial
position, goal, yaw, heading, root velocity, joint state and scan. Use the frozen
verifier's existing tolerances. Recompute each outcome from raw CSV with
`verify_trial`; require final native acknowledgement, no collision, and both
strict geometric thresholds. Retain all failures and disagreements.

Then run the frozen `validation.py analyze SNAPSHOT OUTPUT`. It repeats the
checks and computes the single registered decision. Let W be corrected-only
successes and L be original-only successes. Success gain is `(W - L) / 20`.
The two-sided exact paired probability is
`min(1, 2 * sum(comb(W + L, k), k=0..min(W,L)) / 2**(W + L))`.
For zero discordant pairs the probability is 1. All four conditions must pass:

- Success gain is at least 0.10.
- Exact paired probability is at most 0.05.
- No pair has a corrected collision without an original collision.
- Corrected mean planning time is at most 0.20 s. This is the unweighted mean of
  the 20 per-trial means, with each trial mean taken across its recorded commands.

Report both success counts, W, L, gain, probability, collisions in both roles,
added-collision pairs and planning time. List each lost original success as a
regression. The fixed primary rule permits losses if every condition still
passes; do not silently replace it with an all-successes-preserved endpoint.
Path length, steps and other diagnostics are descriptive only. If the primary
rule fails, report failure. There is no alternate endpoint, failed-layout
replacement, post-outcome sample expansion or tuning on these cases. A launch
or verification fault leaves the study incomplete; retain its artifacts and
report it before any change or rerun. Never reclassify it as a successful study.

## Limits

This study tests the controller correction in transport mode on 20 fixed,
procedurally generated simulator layouts. It does not test JEPA superiority,
real hardware, all terrain types or all reset states. The bounded sample can
miss smaller effects. Development wins selected this controller and cannot be
pooled with fresh pairs. Known successful controls took more commands after the
correction, so blanket travel-time improvement is not assumed. Preserve the
earlier failed 649 gate, rejected heading candidate and original model comparison.
