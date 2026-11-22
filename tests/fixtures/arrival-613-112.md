`arrival-613-112.npz` contains the two recorded arrival states, the last measured
scan, and the unchanged frozen `GoalDynamics` weights. It also contains the prior
layout 501, seed 30 state and its measured zero-command endpoint.

The test replays the measured zero-command motion exactly. For other commands it
adds the frozen model's change in motion relative to zero. This checks command
selection against the observed recoil. It does not prove the native physics.
The policy's prior command is a motion proxy, not a measured velocity.

Sources and SHA256 values are also stored in the NPZ:

| Source | SHA256 |
| --- | --- |
| Confirmation layout 613, seed 112 JEPA CSV, steps 308–309 | `85d3cfe920d9acbb122bcf3e7c45f93f73b906e2e5a143ed9d38dd370ec14fb2` |
| Its saved scan, `last` array (the pre-step-309 scan) | `a6a2166678d87d2b510cbfa2721d709e447ac2d5b697f99afab84df784a62b6f` |
| Original `runs/rover-baseline/20261113T210435968626Z/model.pt` | `76953ac81e383a6000d9b26bf8026b5049f4ce768831bd48506dea31d7649283` |
| Layout 501, seed 30 model CSV, step 224 | `928ac219775877c0255dccb9e3f10981f4a3955cc5cd73648da7e2089dfb5460` |
| `notes/rover-goal-hold-summary.json`, model-hold seed 30 endpoint | `61c3cd33c29c066e6c8a5974932f1728bdabae7df24a7e4d4aa5aca4750aa85a` |

The CSV state fields are the prior speed and turn commands, distance times 0.11,
goal bearing relative to yaw divided by pi, and heading error divided by pi.
Replaying the caller's alignment latch over the full seed 30 CSV gives `True`
at step 224. Both latch values are checked for the translating arrival.
