# Rover JEPA training and data collection

The starting model was **rover-dino-wm-v1**: Meta pretrained DINOv2 ViT-S/14,
kept frozen, with the actual JEPA-WMs `ViTPredictor` trained from random weights.
This adapts the upstream DINO-WM variant to rover height scans. No PointMaze
dynamics weights are transferred. The encoder does not train on rover data.

The current **rover-dino-wm-v6** has 167,904 trainable parameters. A projection
fitted only on training data reduces 384 encoder features to 64 channels.
The Meta predictor has two layers, width 96, four heads, and a residual output.
Evaluation still scores all 384 original features. The projection does not hide
prediction error.

```text
Height scan → fixed height scale → 112 × 112 grid → frozen DINOv2
                                                        ↓
Past two feature grids + commands → motion alignment → Meta predictor
                                      ↑                     ↓
                         Frozen motion baseline       Feature correction
                                                            ↓
                                              Future terrain features
```

The encoder produces 64 spatial tokens. Each scan uses three identical image
channels and a fixed ±2 m height scale. There are no RGB frames. Two context
observations are one second apart. Five commands are applied each second.
Forecasts run forward using predicted observations and state, without feeding
recorded future state into the model.

The existing GoalDynamics baseline trains first on the same recording split.
It predicts forward, lateral, and yaw motion. Its weights then stay frozen.
Exact coordinate transforms update goal pose and align both past feature maps.
The JEPA predictor learns terrain corrections. Pose gains come from the motion
baseline, not from a new JEPA pose head. Lateral motion remains free; no no-slip
constraint is imposed. The raster extent is 5.05 m, matching the sample centers
and the interpolation convention.

Pinned sources:

- JEPA-WMs: `13cf1d9c7e476f53c17714d2e0f1dc239a883ce0`.
- DINOv2: `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- Encoder weights SHA256: `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

## Data and jobs

| Split | Recordings | Raw rows |
| :--- | :--- | ---: |
| Original training | Mars 0–3; debug 3 | 4,200 |
| Added training | Mars 16,18; debug 17,19 | 12,000 |
| Development validation | Debug 1 | 900 |
| Development test | Debug 2 | 900 |
| Reserved final test | Mars 20; debug 21 | 6,000 |

The collection Job completed all six new recordings: 18,000 rows. It runs
RLRoverLab with scripted forward, reverse, turns, and stops. It records data;
it does not train an RL policy. Every file has a completion marker. The split
preparer accepts only completed training files and snapshots their paths.
Seeds 20 and 21 stay reserved under the
test plan saved with the run artifacts outside Git.

The reused episode loader checks finite values, timing, reset boundaries, and
stale terminal fields. Forecast windows cannot cross resets or recordings.
Only the documented original seed-0 path can omit its legacy metadata file.
Development recordings have been inspected repeatedly. They are not fresh tests.
All recordings use two existing terrain layouts.

`train-job.yaml` runs the baseline, then the JEPA predictor. Every epoch checks
one- and five-second forecasts. The training objective normalizes each horizon
by training-only copy error. Each horizon receives equal weight after normalization.

Training stops at a validation plateau: less than 0.1% cumulative improvement
for eight checks. Each new best checkpoint is saved, even below that threshold.
The maximum is 60 epochs. A quality pass can stop training after five epochs.
No workload has an elapsed-time cutoff. Do not stop a running job based on a
compute estimate. Cluster admission adds cleanup of completed Job metadata;
all artifacts stay on the shared volume.

## Fixed quality target

Both development splits must pass every check at both one and five seconds:

- Feature error at least 10% below copying current features.
- Feature error at least 2% below motion-only transport.
- Shuffled future commands increase feature error by more than 2%.
- Pose error at least 10% below copying current pose.
- Shuffled future commands increase pose error by more than 2%.

The shuffle preserves past commands and permutes only future command blocks
across the full split. `FINISHED` means execution completed. Only
`QUALITY_PASSED` means the complete target passed. Never weaken these gates to
make a run pass.

Final offline findings are consolidated in [results](../../docs/results.md#rover-jepa-prediction).
Keep development attempts, metrics, and histories with the run artifacts
outside Git.

## Saved evidence and final evaluation

Artifacts persist under `/data/runs/rover-jepa/<pod>/`: source snapshot,
`split.json`, hardware, console log, exit code, baseline checkpoints,
`training/manifest.json`, `history.jsonl`, `best.pt`, and `metrics.json`.
The manifest records source and data hashes. Encoder weights remain under
`/data/cache/torch`; pinned encoder code is required to use the checkpoint.
Collection files persist under `/data/datasets/rover-jepa-collection/`.

`evaluate.sh` and `scripts/evaluate_rover_jepa.py` evaluate the reserved data.
They require a development pass, verify frozen source and encoder hashes,
load the selected checkpoint strictly, and replay validation before opening
reserved observations. Final reports contain pooled and per-terrain scores.
No fitting occurs on final data. Do not use final scores for further tuning.

These are offline prediction checks. They do not establish closed-loop rover
navigation, real hardware performance, wheel-slip estimation, or safety.
