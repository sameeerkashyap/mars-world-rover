# Completed rover predictor pilot

## Decision

Keep the current predictor and motion-prior navigation path. Do not replace it
with the tested AdaLN or unwarped DINO-WM model. All six GPU arms and both CPU
reference arms failed the offline candidate gate. The stronger independent
evaluation reached the same conclusion.

This conclusion applies to this small development experiment. It does not
establish that one architecture is best across rover datasets or tasks.

## What trained

The encoder was frozen DINOv2-S/14. Both new predictors used the same full
384-value features at 112 pixels, two observed frames one second apart, and
five actual 0.2-second commands per prediction interval. Neither received
motion-transported features or future measured state.

Each predictor had two transformer layers and four heads. DINO-WM had
4,855,712 trainable parameters; AdaLN had 6,076,128. This compares two
implementations, not equal parameter counts. The current 167,904-parameter
residual predictor had a different earlier training history and a 64-value
PCA representation. It is a deployment reference, not a matched training arm.

The pilot sampled 256 training windows from nine recordings and 128 validation
windows from the existing debug-seed-1 recording. Scaling used training data
only. Training used AdamW at 0.0003, batch 16, gradient clipping at 1, weight
decay 0.0001, two predicted intervals, and detached predicted inputs. The
reduced-context loss supervises last-context predictions at t+2 and t+3;
it is an explicit rover adaptation, not the complete upstream benchmark recipe.

Seeds 0, 1, and 2 ran on an NVIDIA A10 with PyTorch 2.10.0+cu128. The 60-epoch
maximum was not reached: validation stopping ended the six arms after 19–32
epochs. Each passed the tiny-window learning check and checkpoint reload.
The separate CPU reference used seed 0 and stopped after 23/20 epochs for
DINO-WM/AdaLN. No job was stopped because of elapsed compute time.

## Results on the same 128 validation windows

Errors are mean squared errors in the original DINO feature scale. New-model
rows show medians over three initialization seeds. The controls are recomputed
on exactly the same windows. These absolute errors use a different scale from
the older normalized-feature training reports and must not be mixed with them.

| Forecast | 1-second error | 5-second error |
| --- | ---: | ---: |
| Copy current features | 0.895281 | 2.212706 |
| Full-feature transport, full-data motion prior | **0.776731** | **1.640161** |
| Current transported-residual predictor | 0.793877 | 1.704415 |
| Current model's matching PCA transport | 0.809197 | 1.766347 |
| New DINO-WM | 0.855549 | 1.837537 |
| New AdaLN | 1.097972 | 1.979854 |

The DINO-WM median was about 10% worse than full-feature transport at one second
and 12% worse at five seconds. AdaLN was about 41% and 21% worse. Neither passed
in any seed. The current residual still improves over its own matching
transport control. The two controls use separate motion checkpoints whose
corresponding tensors differ by at most 0.000000179. The full-feature control
moves all 384 features; the PCA control persists components outside its
64-value subspace. Their errors therefore do not isolate learned correction
quality alone. Lower full-feature error does not establish better predictions
from the existing 64-channel terrain-cost readout.

The central 2-by-2 patches also favor transport:

| Forecast | Central 1-second error | Central 5-second error |
| --- | ---: | ---: |
| Full-feature transport | **0.652195** | **1.446616** |
| Current residual | 0.655509 | 1.458008 |
| New DINO-WM, median | 0.742511 | 1.715972 |
| New AdaLN, median | 0.991683 | 1.835988 |

These are feature errors near the rover. They are not terrain-cost errors,
collision probabilities, or measured navigation outcomes.

## Verification and interpretation

Review found weak adjacent-window action shuffling, missing grouped errors,
a missing current-model control, and missing selected-weight digests. Revision 2
corrects these and saves cross-episode donor indices. It also rejects reordered
validation recordings, which would otherwise change positional window indices.
The [review record](world-model-pilot-review.md) describes the corrections.

All six copied checkpoint hashes matched the original shared-volume files.
Independent CPU re-evaluation changed the original GPU feature errors by at
most 0.00000251, far below the measured deficits. The completed CPU reference
also rejected both predictors. Its stronger re-evaluation confirmed that result.

Training error fell while validation improvement stopped. The new validation
error on the normalized two-interval training objective was about 0.438–0.445
for DINO-WM and 0.507–0.523 for AdaLN. More epochs or a larger predictor are not
justified by this pilot. Before spending that budget, test data diversity,
state observability, and loss/representation choices against the strong motion
reference. A changed model still needs a valid readout and simulator evidence.

Only one existing debug layout supplied validation, and windows overlap.
Initialization seeds are not independent terrain samples. No reserved final
recording was opened, and no new predictor controlled the rover. These results
support a decision to retain the current approach, not a generalization claim
or a globally optimal hyperparameter setting.

## Artifacts

- [Stronger GPU checkpoint evaluation](world-model-pilot-gpu-evaluation-v2.json).
- [Compact GPU summary and transfer check](world-model-pilot-gpu-summary.json).
- [Stronger CPU-reference evaluation](world-model-pilot-cpu-evaluation-v2.json).
- [Original GPU report](world-model-pilot-v1-gpu.json) and [CPU report](world-model-pilot-v1-cpu.json).
- [Model implementation PR](https://github.com/sameeerkashyap/mars-world-rover/pull/10).

Original GPU artifacts remain at
`/data/runs/rover-world-model-pilot/mars-rover-world-model-pilot-20261121-v1/`.
CPU artifacts remain under the corresponding
`mars-rover-world-model-cpu-check-20261121-v1/` directory. Its revision-2 report
is in `evaluation-v2/`. Raw reports were preserved. Evaluation source and
checkpoint hashes are recorded with the new metrics.

## Navigation follow-up

The separate goal-hold replay tests whether stopping once both measured goal
tolerances are met avoids terminal drift while waiting for the native flag. It uses
the original motion checkpoint and inspected terrain, with unchanged success
criteria. It completed six paired trials: the legacy planner reached 2/3 goals
and goal hold reached 3/3, with zero collisions in both groups. Seed 30 ended
at 0.178301 m with hold, versus 0.180314 m without it. The same hold behavior is
now the planner default, with an explicit legacy replay path. See the
[verified replay](rover-goal-hold-verification.json) and
[summary with frozen source hashes](rover-goal-hold-summary.json).

This fixes an observed control boundary case on inspected terrain. It does not
establish a learned-model gain or performance on new layouts.
The control change and its replay are in the
[navigation PR](https://github.com/sameeerkashyap/mars-world-rover/pull/11).
