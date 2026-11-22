# Arrival correction: native development replay

The layout 613, seed 112 JEPA arrival now passes the unchanged native and strict
geometry checks. The original and corrected measured states and commands match
exactly through step 308. At step 309, the command changes from zero to
`[0.05, 0.04]`. The original endpoint was 0.180284768 m and 0.082674026 rad;
the corrected endpoint is **0.176925346 m and 0.080507994 rad**. Both native
flags are true, but only the corrected endpoint is inside distance < 0.18 m
and absolute heading error < 0.1 rad. Neither trace has a collision.

The [last three old/new rows](rover-arrival-native-boundary.csv) show the boundary
change. [The receipt and all 18 endpoints](rover-arrival-native-validation.json)
include raw CSV hashes and independent `verify_trial` checks.

| Known case | Controller | Final distance (m) | Heading error (rad) | Strict native success |
| --- | --- | ---: | ---: | --- |
| 613 / 112 | JEPA | 0.176925346 | 0.080507994 | yes |
| 613 / 112 | transport | 0.161531955 | 0.052302122 | yes |
| 501 / 30 | model-hold | 0.178300649 | -0.078181028 | yes |
| 501 / 31 | model-hold | 0.172808826 | -0.085341215 | yes |
| 501 / 32 | model-hold | 0.168159455 | -0.078771591 | yes |
| 501 / 30 | unheld model | 0.180314064 | -0.057119370 | no |

The earlier turning-only hold regression is preserved. The unheld seed 30 miss
also remains reproducible. Both successful layout 601 controls still pass.
Their complete measured-state/command traces match the original traces exactly
(transport: 304 steps; JEPA: 318 steps). The layout 613 transport trace also
matches all 598 original steps. The four shared-timeout layouts remain failures
in both arms. All 18 traces have zero collisions; nine pass strict native checks.

The completed A6000 batch returned exit code 0. It used immutable source
`68ff236449953775eacc1b3e178f5f167df589fd`, Isaac Sim 6.0.1.0 and Torch
2.10.0+cu128. All 12 snapshot checksum entries and 35 copied case-source files
were checked. Direct PVC hashes of motion, JEPA, terrain readout, encoder weights
and predictor source match the frozen inputs. The planner hash also matches
integration `4eedd3c`; later runner telemetry changes are outside this replay.
The runtime receipt reports `pass` and preserves its separate `pip_check` exit 1.

Durable session root:
`/data/runs/rover-controller-session/mars-rover-controller-session-a6000-20261122-v1-9gnkn/`.
The run is `development-68ff236-v2/`; source and plans are under
`snapshots/68ff23644995/`. The launcher, exit code, source check, runtime and
hardware receipts remain there. Full CSV, JSON, scan and pre-command HDF data
remain in each case directory. The historical confirmation run is unchanged.

Initial pose, goal, heading and scan match the original layout 613 and 601
inputs. This batch omitted initial velocity and joint receipts, and the local
historical export lacks generated meshes for an independent mesh comparison.
The separate matched replay will check full initial state and mesh equality.
These are inspected development cases. Sustained parking after native termination
and new-layout performance remain untested. No fresh model-benefit claim follows.

Recheck the exported traces from the repository root with the existing public
CSV verifier. This reads measurements; it does not run physics:

```bash
rtk proxy python3 - /private/tmp/rover-controller/native-dev <<'PY'
import importlib.util
import json
from pathlib import Path
import sys

path = Path("deploy/nautilus-jepa-navigation/verify.py")
spec = importlib.util.spec_from_file_location("verify_navigation", path)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)
results = []
for path in sorted(Path(sys.argv[1]).glob("*/*/summary.json")):
    summary = json.loads(path.read_text())
    results.extend(verifier.verify_trial(path.parent, trial) for trial in summary["trials"])
assert len(results) == 18 and not any(row["collision"] for row in results)
print(len(results), "traces verified;", sum(row["verified_success"] for row in results), "strict native successes")
PY
```
