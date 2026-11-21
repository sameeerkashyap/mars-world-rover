# AdaLN versus DINO-WM pilot

This trains two predictors on the same rover recordings. It does not replace
the deployed navigation controller. DINOv2 stays frozen. The learned arms use
full 384-value terrain features at 112 pixels, without motion transport or goal
fields. `src/rover_world_model.py` contains the model adapter. It loads the
pinned upstream classes rather than a copied transformer implementation.

Both arms use two context frames, five actual 0.2-second commands per prediction
interval, two transformer layers, four attention heads, two predicted training
intervals, AdamW at 0.0003, batch size 16, and three initialization seeds.
DINO-WM has 4,855,712 trainable parameters; AdaLN has 6,076,128. The comparison
is not parameter-count matched. AdaLN receives previous-command features
expanded across patches because its upstream feature-conditioning path expects
that layout.

The initial study samples 256 training windows and 128 validation windows at
fixed, evenly spaced indices. It reuses the nine training recordings and one
development validation recording. Normalization is fitted on all training
recordings only. No reserved test recording is encoded or scored. The saved
split lists the older development test file for provenance, but the trainer
does not consume it.

Each arm first checks learning on up to 64 training windows with 100 optimizer
updates. Its final loss must be below half its initial loss. The predictor and
optimizer are then reinitialized before comparison training. Training uses the
last-context predictions at t+2 and t+3 with detached predicted inputs; this is
an explicit reduced-context adaptation, not a full upstream benchmark run.

Maximum training length is 60 epochs, with validation stopping after eight
epochs without a material improvement. The best checkpoint minimizes mean
one-/five-second validation error relative to transport. The saved candidate
gate requires at least 2% lower error than transport at both horizons and
increased error when future action blocks are shuffled. This is a screening
gate, not statistical evidence or automatic promotion. Report each seed.

Copy and frozen motion transport use the same full-feature space. Errors are
measured on the original feature scale. The motion prior is the completed
scaled run's baseline and its hash is recorded. It is a control only; no pose
or transported feature enters the learned predictor.

Launch once:

```bash
rtk proxy bash deploy/nautilus-world-model-pilot/deploy.sh
rtk kubectl --context nautilus -n razvanlab get pods -l job-name=mars-rover-world-model-pilot-20261121-v1
rtk kubectl --context nautilus -n razvanlab logs job/mars-rover-world-model-pilot-20261121-v1 -c trainer
```

The deployment freezes its source ConfigMap and refuses to replace an existing
job. It has no elapsed-time limit or automatic timed deletion. It shares the
existing PVC and feature cache. Results persist under
`/data/runs/rover-world-model-pilot/mars-rover-world-model-pilot-20261121-v1/`.
Each arm saves source/data hashes, sampled indices, normalization, fit check,
best weights, optimizer/RNG resume state, and metrics. `comparison.json`
combines the six arms. `FINISHED` means execution completed, not a quality pass.
Re-execution with identical source/config resumes saved epoch boundaries.

After the pilot, inspect per-seed gains and failure cases before changing
learning rate, context, capacity, or resolution. A useful offline result still
needs a validated terrain readout and paired Isaac Sim navigation trials.
See the [source decision](https://github.com/sameeerkashyap/mars-world-rover/issues/6)
and [Wayfinder research plan](https://github.com/sameeerkashyap/mars-world-rover/issues/4).
