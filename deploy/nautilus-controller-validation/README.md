# Controller replay and comparison tooling

This tooling prepares source snapshots. It does not submit cluster jobs, train a
model, or select the final controller. Use the existing pinned navigation setup
and session. Do not change an executing snapshot.

`validation.py prepare REF SNAPSHOT --checkpoint MODEL --predictor-source VIT`
reads a committed source tree. It pins the original planner at `815f19e`, uses
`REF` for the corrected planner and the common runner, and copies all required
local sources with SHA256 receipts. The snapshot directory must be new. Files
are read-only. Keep `SHA256SUMS` with the upload receipt and validate it after
upload. Keep the snapshot and its plan with the evidence.

The default phase is `development`, and the default mode is `transport`.
`--mode jepa` prepares 613/112 and successful control 601/100. `transport`
prepares 604/103, 611/110, 614/113, 615/114, and successful control 601/100.
`--mode model-hold` prepares scan-aware model hold on 501/30–32.
These are inspected development cases. Their results cannot establish a fresh
comparison endpoint.

Only after the native development findings are resolved, use `--phase fresh`
to freeze the final source and one controller mode. The design fixes layouts
701–720, seeds 200–219, one original/corrected pair per layout, and alternating
planner order. Both versions use the same mode, weights, runner, terrain, native
goals, and safety checks. Do not run extra cases to replace a failed endpoint.
The [fresh v1 protocol](fresh-v1-protocol.md) registers the selected controller,
exact snapshot hashes, transport mode, case order and analysis rules before
outcomes. Its generated plan and snapshot manifest are committed beside it.
The [fresh v1 results](../../notes/controller-fresh-native-validation.md) record
the completed study and failed primary decision.

Inside the prepared simulator runtime, with the existing stack and hardware
receipt saved as JSON, run:

```bash
rtk proxy python SNAPSHOT/validation.py run SNAPSHOT OUTPUT \
  --runtime-receipt SESSION_RUNTIME_JSON \
  --jepa-run FROZEN_V12_RUN --terrain-checkpoint FROZEN_READOUT
```

Omit the forecast arguments only for `model-hold`. `SNAPSHOT` and `OUTPUT` must
be absolute paths. The tool sets the snapshot Python source path. It uses the
current interpreter and the existing simulator setup. It writes a new output
directory, copies the runtime receipt, records execution order, and invokes the
public native runner once per planner and case. The runner checks the selected
planner hash before import. It saves the actual role, source hash, copied source,
full initial state, scan, terminal trace, and pre-command HDF telemetry. Original
planner diagnostics are marked unavailable; its command call stays compatible.
The tool does not impose a job deadline or stop a workload for a compute budget.

Run the independent checks before analysis:

```bash
rtk proxy python SNAPSHOT/validation.py verify SNAPSHOT OUTPUT
rtk proxy python SNAPSHOT/validation.py analyze SNAPSHOT OUTPUT
```

Verification checks source hashes, registered cases, simulator receipts, matched
initial pose/goal/heading/velocity/joints/scans, exact generated meshes, and native
CSV truth through `verify_trial`. It does not replay physics. Analysis reruns
verification and rejects development evidence. The declared fresh endpoint uses
native success gain of at least 0.10, a two-sided exact paired test at 0.05,
no added collision in a pair, and corrected mean planning time at most 0.20 s.
The report includes wins, losses, gain, exact probability, collisions, and
latency. It has no conditional alternate endpoint.

Old test plans, frozen evidence, and their source snapshots remain separate.
