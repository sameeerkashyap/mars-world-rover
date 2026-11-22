# Existing JEPA terrain forecasts: completed navigation comparison

The current frozen v12 JEPA terrain forecasts did not establish better goal
completion than matching motion-only transport. Keep transport as the working
forecast baseline. The learned correction remains an experimental option.

Forty native Isaac Sim trials covered twenty new independent rock layouts,
601–620, with one matched trial seed per layout, 100–119. The controllers shared
the motion model, 64-value terrain representation, terrain readout, candidate
commands, goal-arrival hold, measured-scan rejection and success thresholds.
Only the learned feature correction differed. Controller order alternated.
No model was trained or tuned during this comparison.

## Result

| Measure | Matching transport | Existing JEPA |
| --- | ---: | ---: |
| Strict goals completed | **16/20** | **15/20** |
| Native success flags | 16/20 | 16/20 |
| Collisions | 0 | 0 |
| Native episode timeouts | 4 | 4 |
| Stationary-command stalls | 0 | 0 |
| Mean trial planning time | **108.40 ms** | **130.13 ms** |
| Per-command planning p95 | 134.21 ms | 153.84 ms |
| Longest measured planning call | 0.874 s | 3.033 s |

Strict success requires native acknowledgement, terminal distance <0.18 m,
heading error <0.1 rad, and no collision. Fifteen pairs succeeded in both arms,
four failed in both, and one succeeded only with transport. There were zero
JEPA-only successes. The observed success difference is -5 percentage points;
the exact two-sided paired discordance test gives p = 1.0. This does not support
a reliable improvement or a reliable harm claim from one discordant pair.
The preregistered promotion rule failed. Mean JEPA planning was 20.05% slower.

The rule required a success gain of at least ten points, exact p <=0.05, no added
collision per case, and mean JEPA planning <=0.2 s. All twenty pairs were retained.
There was no interim success-rate test, outcome-based replacement, or threshold
change. Means below 0.2 s do not establish a hard real-time bound; the measured
maximum calls exceeded that interval.

On the fifteen common-success pairs, JEPA paths averaged 9.13459 m versus
9.18447 m for transport, a 0.54% reduction. Simulated completion time averaged
65.60 s versus 73.49 s, a 10.74% reduction. These are descriptive results on a
subset that excludes five pairs. They do not replace the failed primary endpoint
and are not a confirmed efficiency claim. The mean path across every trial is
not an efficiency measure because failed trials can travel shorter distances.

The stationary-stall measure means at least 25 consecutive exactly zero
commands. It does not detect every form of failed progress. Both controllers
timed out on layouts 604, 611, 614 and 615, despite this stall count being zero.

## The forecasts changed commands

The runner evaluated the other forecast mode on the same actual history after
each scored action. It did not execute the diagnostic action. On transport
histories, choices differed on 1,293 of 8,568 audited steps, or 15.09%. On JEPA
histories, choices differed on 1,146 of 7,695 steps, or 14.89%. These correlated
steps are a mechanism audit, not independent statistical samples. Diagnostic
work was excluded from selected-controller latency and its counters were restored.

The learned correction reached the command selection step. Its action changes
did not supply a supported success gain in this experiment. Candidate forecasts
still stop at five seconds, while some turn-then-drive candidates extend to
sixteen seconds. This is a limit of the frozen integration, not a demonstrated
cause of the result. The measured scan still rejects unsafe or unknown paths.

## Remaining goal-arrival counterexample

Layout 613, seed 112, is the single transport-only success. Transport terminated
at distance 0.161532 m and heading error 0.052302 rad. The JEPA run entered both
geometric tolerances at step 308: distance 0.178692 m and heading 0.081869 rad.
At step 309 it issued the goal-hold zero command. During that step, distance
increased by 1.593 mm to 0.180285 m while the native success flag became true.
The endpoint is 0.285 mm outside the strict radius and remains a failure.

This shows that the earlier known-layout goal-hold fix does not cover every
arrival. It does not demonstrate that the learned terrain representation is
worse. Preserve this case for a braking/arrival diagnosis. Do not relax the
threshold or relabel the run. Any correction is new development work and cannot
change this frozen result. [Final trace excerpt](jepa-arrival-boundary-case.csv).

## Integrity and reproduction

All twenty pairs had exactly equal measured initial position, goal, yaw,
heading error, root velocity, joint position, joint velocity and height scan.
All fourteen visible source files matched their frozen hashes. Every raw CSV
passed the corrected outcome, command, heading, path-length and audit checks.
All twenty generated meshes matched independent hashes committed before results.

- Controller/runtime source: `e1ba34c`; original [frozen plan](../deploy/nautilus-jepa-confirmation/frozen-v1-test-plan.json).
- Corrected analysis checker: `737f77c`, SHA256 `83d71101bbba5bdee7ff07e67a6622fb3bfe4e4357f94b02e2a555ae1a15a774`.
- [Full verified outcomes](jepa-navigation-confirmation-verification.json).
- [Paired analysis, latency, state checks and artifact receipt](jepa-navigation-confirmation-analysis.json).
- [Independent mesh hashes](jepa-confirmation-mesh-hashes.json).
- [Protocol and analysis correction](../deploy/nautilus-jepa-confirmation/README.md).

Nautilus job `mars-rover-jepa-confirmation-20261122-v1` ran all forty simulations
from 02:13:18 to 03:59:39 UTC on 2026-11-22, on one NVIDIA A10. The image digest was
`sha256:6c2d59d37023f45e990aa7f8237c54e2c483cb376b48280bb4da7d892dd75fff`.
The runtime used Isaac Sim 6.0.1.0, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128,
NumPy 2.3.1, h5py 3.15.1 and einops 0.8.1. The driver was 615.71.09. Full package
lists, raw traces, scan files, source and mesh arrays remain on the shared volume:

```text
/data/runs/rover-jepa-confirmation/mars-rover-jepa-confirmation-20261122-v1-rg5hp/
```

The job exited 1 only after simulation completion: its original checker mistook
Kubernetes hidden ConfigMap copies for duplicate source files. That issue was
identified and the correction published during execution, before the primary
comparison was opened. The running source remained immutable. The correction
checks visible source bytes, retains valid zero-scoring trials, and checks initial
heading explicitly. The source and primary statistical test for the controllers
did not change. [Recorded amendment](https://github.com/sameeerkashyap/mars-world-rover/issues/13#issuecomment-7073353689).

A separate read-only CPU job exported the complete raw run. Independent local
reanalysis with the corrected checker passed all forty traces. The 3,981,065-byte
archive has SHA256 `917fef92ac94cd8e4924e46070c70dc8a1ab28a213be89d52aed2bf683089c5b`.
The raw job's failed packaging status is preserved in the artifact receipt.
No running workload was stopped, and no PVC data was deleted.

Checks: the original 61-test suite passed. After the analysis correction, all
five focused confirmation tests passed, including source-copy and scan-tampering
fixtures. The corrected checker was also run against the actual exported data.

This study covers one synthetic terrain family. The sample cannot rule out a
smaller benefit elsewhere. The twenty inspected layouts are now development
data. Broader terrain, friction, and a new efficiency endpoint need a separate
frozen experiment. No larger model-training campaign is justified by this result.
