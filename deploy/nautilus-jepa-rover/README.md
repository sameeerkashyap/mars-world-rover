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
[test plan](../../docs/results/rover-jepa/test-plan.json).

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

Run v11 stopped after 17 epochs and selected epoch 16. It passed every check
except the one-second validation gain over transport: 1.83%, below 2%.
It improved over copying by 10.41% at one second and 21.83% at five seconds on
validation. The five-second development-test gain over transport was 5.22%.
Run v12 repeated this source and data. It passed every development check at
epoch 23 and stopped. GPU numerical differences changed the training path;
this is evidence of a narrow margin, not a large robust improvement. The
reserved final evaluation used its frozen checkpoint and passed the planned
pooled target. Final feature gains over copying were 10.80% at one second and
19.88% at five seconds. Gains over transport were 2.37% and 5.65%. Mars alone
narrowly missed the one-second copy target at 9.85%; debug passed. See the
[final report](../../docs/results/rover-jepa/final-metrics.json) and
[results](../../docs/results.md). All workloads have completed.

## Train terrain costs from the recordings

`src/terrain_cost.py` adds a 65-parameter readout from each 64-channel spatial
feature to a terrain cost between zero and one. The target is the fraction of
each patch blocked by the existing height-scan clearance rules. Scan resizing
uses the same 112-pixel grid and 14-pixel patches as the encoder. DINOv2,
the selected JEPA predictor, its projection, and its motion prior stay frozen.

Run the training script as a module from the repository root. It accepts CPU
or CUDA. The encoder source must be the pinned DINOv2 snapshot listed above.
Use its existing weight cache through `TORCH_HOME` when available.

```bash
rtk proxy uv run --no-project --python 3.12 \
  --with torch==2.10.0 --with torchvision==0.25.0 \
  --with numpy==2.3.1 --with h5py==3.15.1 --with einops==0.8.1 \
  python -m scripts.train_terrain_cost \
  --run runs/rover-jepa/attempt-10 \
  --data data/rover/rover_seed_0.hdf5 \
    data/rover/mars-rover-variation-0/rover_seed_1.hdf5 \
  --validation-data data/rover/mars-rover-variation-1/rover_seed_1.hdf5 \
  --encoder-source /runtime/dinov2 \
  --predictor-source external/jepa-wms/app/plan_common/models/vit.py \
  --output runs/terrain-cost/new-run --epochs 60 --device cpu
```

Replace `/runtime/dinov2` with the local pinned source path on a workstation.
The example uses the two training recordings available locally. A full run
can list all original training files. The script checks file content hashes
against the selected run's `split.json` and checkpoint manifest. Validation
files cannot enter training. Development-test and reserved-test files are
rejected, including renamed copies. Windows stay inside their original episodes.

The best epoch is selected by measured validation cost error. `metrics.json`
also compares one- and five-second cost forecasts from JEPA, motion-only
transport, copying current features, and the training mean for each patch.
All three feature forecasts use the same readout. `best.pt` stores its weights,
the fixed projection, normalization, and the source JEPA checkpoint hash.
Reloaded predictions must match before `FINISHED` is written. Output folders
must be new; existing artifacts are never replaced.

These labels express geometric clearance rules. They do not measure traction,
wheel slip, or physical safety. `FINISHED` means execution completed, not that
JEPA improved navigation. Online planner integration remains a separate step.

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
