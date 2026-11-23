# Fresh controller comparison: primary rule failed

All 40 registered native trials completed and passed independent host verification.
The corrected controller succeeded on 17/20 new layouts (85%), versus 13/20 (65%)
for the original. There were four corrected-only successes, no lost original
successes and zero collisions in either role. The registered primary rule
**failed**: the two-sided exact paired probability was 0.125, above 0.05.
Collection and verification are complete; the primary target was not met.

The [frozen protocol](../deploy/nautilus-controller-validation/fresh-v1-protocol.md)
was merged and pushed as `710f24a16ea2c864561d96689a32b5289beebb93`.
Its [registration comment](https://github.com/sameeerkashyap/mars-world-rover/issues/18#issuecomment-7086964853)
was posted at 2026-11-22 22:38:09 UTC, before the launcher started at 22:39:45 UTC.
The study kept layouts 701–720, reset seeds 200–219, alternating planner order,
transport mode, frozen weights and the selected source unchanged. No case was
replaced, rerun or added. No alternate endpoint was used.

## Registered decision

The [compact receipt](controller-fresh-native-validation.json) contains all 20
pairs, both roles' terminal outcomes and CSV hashes, the analysis and source,
runtime, launch and export receipts. The frozen analyzer gives:

| Required condition | Observed | Decision |
| --- | ---: | --- |
| Success gain ≥0.10 | 0.20 | Pass |
| Two-sided exact paired probability ≤0.05 | 0.125 | **Fail** |
| Added collision pairs =0 | 0 | Pass |
| Corrected mean planning time ≤0.20 s | 0.053361405 s | Pass |

Planning time is the unweighted mean of the 20 per-trial means. Four pairs changed
from failure to success: 701/200, 712/211, 714/213 and 715/214. No pair changed
from success to failure. Both controllers timed out on 702/201, 708/207 and
717/216. All 40 trials had zero collisions. Success requires final native
acknowledgement, distance <0.18 m, absolute heading error <0.1 rad and no collision.

## All registered pairs

Commands and path lengths are descriptive. The table gives strict native outcomes
and the corrected controller's final geometry. A timeout remains a failure.

| Layout/seed | Original commands/outcome | Corrected commands/outcome | Corrected distance m | Corrected absolute heading rad |
| --- | --- | --- | ---: | ---: |
| 701/200 | 745 timeout | 326 pass | 0.148321 | 0.078245 |
| 702/201 | 745 timeout | 745 timeout | 3.682556 | 2.291414 |
| 703/202 | 352 pass | 327 pass | 0.129951 | 0.066358 |
| 704/203 | 330 pass | 350 pass | 0.171531 | 0.084836 |
| 705/204 | 305 pass | 726 pass | 0.158577 | 0.061912 |
| 706/205 | 294 pass | 324 pass | 0.157604 | 0.057597 |
| 707/206 | 294 pass | 296 pass | 0.149990 | 0.083236 |
| 708/207 | 745 timeout | 745 timeout | 5.492126 | 1.683958 |
| 709/208 | 408 pass | 466 pass | 0.173216 | 0.096712 |
| 710/209 | 294 pass | 365 pass | 0.143242 | 0.077041 |
| 711/210 | 293 pass | 285 pass | 0.173587 | 0.055259 |
| 712/211 | 745 timeout | 561 pass | 0.149474 | 0.092128 |
| 713/212 | 281 pass | 332 pass | 0.144945 | 0.066676 |
| 714/213 | 745 timeout | 416 pass | 0.170783 | 0.056145 |
| 715/214 | 745 timeout | 401 pass | 0.167933 | 0.052625 |
| 716/215 | 320 pass | 369 pass | 0.172922 | 0.055239 |
| 717/216 | 745 timeout | 745 timeout | 2.456422 | 1.443191 |
| 718/217 | 303 pass | 315 pass | 0.154804 | 0.092162 |
| 719/218 | 395 pass | 610 pass | 0.148463 | 0.059922 |
| 720/219 | 358 pass | 362 pass | 0.173185 | 0.054063 |

The correction can cost more commands on successful layouts. For example,
705/204 increased from 305 to 726 commands and 719/218 from 395 to 610. The known
601 control also increased from 304 to 423 in transport mode and 318 to 445 in
JEPA mode in the separate [development study](controller-selection-native-validation.md).
These costs do not change the registered decision.

## Verification and retained artifacts

Both roles used the common runner from selected commit
`aed6e7030de0c0cd17cd9fa0a303e7ec21e6cbe8`; the original planner stayed at
`815f19e8858bd47361a868fb7500e52d2449e76f`. The source snapshot and generated
plan match the registration byte for byte. All 17 payload checks passed after
upload. The live stack and package freeze matched the fixed runtime before launch.

The frozen `verify` and `analyze` commands were independently rerun on the host.
They reproduced the PVC reports exactly. All paired initial positions, goals,
yaws, heading errors, root velocities, joint positions, joint velocities and scan
arrays were exactly equal, including missing scan samples. Every generated mesh
matched the registered terrain arrays. Source/input hashes, execution order,
runtime receipts and raw CSV truth passed all frozen checks. The runtime's pinned
checks pass; its existing nonzero `pip check` result remains recorded.

Durable raw output is under:

`/data/runs/rover-controller-session/mars-rover-controller-session-a6000-20261122-v1-9gnkn/fresh-controller-aed6e70-transport-v1/`

Adjacent folders contain the immutable snapshot
`snapshots/validation-fresh-aed6e7030de0-transport-v1`, launch receipts
`fresh-controller-launch-aed6e70-v1`, and derived export
`fresh-controller-export-aed6e70-v1`. The full raw manifest covers 658 files.
The host export matches all 537 included raw files. Its exact 121 omissions are
40 decision HDF5 files, 80 terrain USD files and the outer console log. Those raw
files remain on the PVC. Root completed the session normally after the verified
local copy: Job Complete, pod Succeeded, rover exit code zero.

| Artifact | SHA256 |
| --- | --- |
| Registered plan | `8b3715ccb5dcf303ad812922d64aea7a6c907585f2ffb8ed4cdd1772ed73482a` |
| Snapshot manifest | `6a5342133f4851d50be9ed8c04f9560c928b20f524be84806ca3749239825666` |
| Full raw manifest | `bb500706c8718975d8f95c38b490cf5a9ca46e12351f076de98173818053e031` |
| Filtered archive, host and PVC | `ac8169d754ec234ed683f3f21693e0a99358d2375df58038abf03da20cb20b8e` |
| Host verification report | `c930f96208e59df8269ee54cd1b613aed5d294a76db130bdfadf03836a93c028` |
| Host analysis report | `12d8cb586258b414c8fac4f114a6a75564702982efd21b55f7c6f5a3974b190a` |

To repeat the host checks, use the saved snapshot and filtered output with
`rtk proxy uv run --no-project --python 3.12 --with numpy==2.3.1 python SNAPSHOT/validation.py verify SNAPSHOT OUTPUT`,
then repeat with `analyze`. Both paths must be absolute. The local export is
`/private/tmp/rover-controller/fresh-controller-summary-export-v1`.

## Limits

This fixed 20-layout simulator study did not meet its registered statistical
target. It provides no JEPA-superiority or real-hardware claim. It compares
controller versions in transport mode with all weights frozen. Known development
cases remain outside fresh inference. The earlier
[failed 649 gate](waypoint-retention-native-validation.md),
[rejected heading candidate](arrival-heading-correction.md) and
[original JEPA comparison](jepa-navigation-confirmation-results.md) remain intact.
No new runtime code, training, extra sample or replacement trial was introduced.
