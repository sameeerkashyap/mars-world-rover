`arrival-heading-501-30-transitions.npz` contains one common pre-command policy
state and measured scan, followed by three recorded action responses from
layout 501, seed 30. The fixture is 29,007 bytes. Its SHA256 is
`9459f04006213d2c471269fbfe08a7d4ba39556fef869c48172699adc0a90ad1`.

| Source capture | SHA256 |
| --- | --- |
| `arrival-heading-501-30-v1.npz`, source `649eb0b` | `713d185078456bd38feac0fbb35772fbdf2aeba7cd25cfcc058e91d7abd0c88a` |
| `arrival-heading-501-30-joint-v1.npz`, rejected source `3d853af` | `9e454ceb7351c9f7c8a8cabb3d594dd62f9c4df8af791433b5c685720d77ab73` |
| `arrival-heading-501-30-brake-v1.npz`, source `faf2a807` | `14ccd976fd27855a6733f1935beb92305e9304bcb4d1e1ecb4b00fa4da528d09` |

Extraction checked each input hash, then asserted
`np.array_equal(first_policy, other_policy, equal_nan=True)` for the last full
policy vector in all three captures. This includes all 10,201 scan values.
Pre-command position and yaw also match exactly. Every policy has SHA256
`8bd7eb98ed3a5ffc9ba3b75ccad1f4c0a0915ad84413cb107a545fffa797fdb7`.
The common state starts at distance 0.172279030 m and heading 0.097599268 rad.

| Actual command: speed, turn | Measured next distance, m | Measured next heading, rad | Native acknowledgement | Strict success |
| --- | --- | --- | --- | --- |
| -0.05, +0.04 | 0.164792120 | 0.101741314 | true | false |
| -0.2, +0.23 | 0.148812354 | 0.118715525 | true | false |
| 0, 0 | 0.168569148 | 0.096337080 | true | true |

All three transitions are collision free. `next_states` columns are world x, y,
z, yaw, distance, and heading error. Values come directly from the final CSV
row. `source_receipts_json` retains each full source capture hash, raw CSV and
HDF hash, original extractor hash, policy hash, and runtime source hashes.
The compact extractor hash is also stored. Original captures and full native
artifacts remain under the existing controller-session evidence roots.

The public `plan_action` test uses the common pre-command vector and frozen
weights from `arrival-613-112.npz`. It finds the chosen command in the table
and checks its measured strict-success outcome. It uses no neural model motion
estimate or residual adjustment for these responses. Only these recorded
responses are covered; this is known-case development evidence.
