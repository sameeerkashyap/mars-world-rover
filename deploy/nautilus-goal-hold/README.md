# Goal-hold development replay

This compares the unchanged scan planner with one small control change: send
zero commands after the measured goal distance and heading are both inside
the native tolerances. The rover waits for the simulator's success flag.
The normal planner now holds at a reached goal. This replay keeps an explicit
legacy controller (`model`) and a hold controller (`model-hold`) so the
comparison stays reproducible.

A rover can drift outside the goal tolerance while waiting for the native
success flag. This replay tests the hold behavior in physics; zero commands
do not guarantee that residual drift stops.

The study runs six trials: `model` and `model-hold` from matched starts at seeds
30, 31, and 32 on the already inspected `test-rocks-501` layout. Both use the
same frozen motion checkpoint, scan rejection, terrain mesh, joint resets,
five settling steps, and native episode limits. Success still requires the
native flag, distance below 0.18 m, heading error below 0.1 rad, and no collision.
No learned forecast or new architecture is used here.

Launch from the repository root:

```bash
rtk proxy python3 deploy/nautilus-goal-hold/deploy.py runs/rover-baseline/20261113T210435968626Z/model.pt
rtk kubectl --context nautilus -n razvanlab get pods -l job-name=mars-rover-goal-hold-20261121-v1
rtk kubectl --context nautilus -n razvanlab logs job/mars-rover-goal-hold-20261121-v1 -c setup
rtk kubectl --context nautilus -n razvanlab logs job/mars-rover-goal-hold-20261121-v1 -c rover
```

The launcher checks the original checkpoint hash and freezes all source hashes
and case settings in an immutable ConfigMap. It refuses to replace an existing
job. There is no elapsed deadline or timed deletion. Source, stack versions,
raw traces, goal-hold counts, summaries, and independent verification persist
under `/data/runs/rover-goal-hold/<pod>/`.

`verify.py` reuses the strict CSV verifier. It checks the paired starts, source
and checkpoint identities, raw success geometry, command bounds, and reported
goal-hold counts. `SUCCESS` means the replay completed and was verified. The
controller comparison determines whether the proposed change helped.

This is a known-layout development check. It cannot establish unseen-layout
generalization or a learned-model advantage. The GPU allow-list includes RTX
cards and the A10 family. A10-family admission follows the RT-core requirement
and documented A10G deployment path; actual support is checked by the job's
runtime stack. See [Isaac Sim 6.0.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/installation/requirements.html)
and [AWS's Isaac Lab GPU compatibility table](https://aws.amazon.com/blogs/machine-learning/scale-robot-reinforcement-learning-with-nvidia-isaac-lab-on-amazon-sagemaker-ai/).

## Result storage

Keep replay summaries, timing diagnostics, frozen source, and verification
reports under `/data/runs/rover-goal-hold/` or ignored local run storage.
Development replay outputs remain outside Git. Only consolidated final
findings belong in [results](../../docs/results.md).
