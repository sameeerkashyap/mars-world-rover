# Global rock clearance diagnosis

All four layouts have geometrically admissible targets and complete conservative planar routes. This rejects simple rock-geometry target or route impossibility for these inputs. It does not establish that the frozen controller can execute the routes or obtain native success. [Ticket 17](https://github.com/sameeerkashyap/mars-world-rover/issues/17) remains open for native diagnosis.

This work is diagnostic only. No geometry, target, path or clearance value was supplied to the navigation controller. No navigation source, model, native criterion or running job was changed for this probe. The reproducible CLI is `scripts/diagnose_rover_clearance.py`. Its [numerical report](rover-global-clearance-analysis.json) includes source and input hashes and every route vertex.

## Prediction before probes

1. If target clearance is the cause, every pose in the strict 0.18 m goal disk at heading error below 0.1 rad will intersect the fixed footprint or be blocked by the unchanged scan mask.
2. If route separation is the cause, an admissible target may exist but no route will connect it to the initial position through centers with adequate rock clearance.
3. If a route exists beyond the local scan, a conservative full-field route can connect the initial pose to an admissible endpoint. This would reject simple geometric obstruction while leaving dynamics and local planning unresolved.

## Inputs and provenance

The source is `src/evaluation_terrain.py`, function `heldout_meshes`, at integration base `4f59f6f884c5af38c6adf81b03fe654e09c2cd1f`. Its generated ground, face, rock and rock-face array bytes were SHA256 checked against `notes/jepa-confirmation-mesh-hashes.json` for every layout. All four arrays matched for every case. The inputs are from the completed frozen run `/data/runs/rover-jepa-confirmation/mars-rover-jepa-confirmation-20261122-v1-rg5hp/`. Existing seeded geometry was regenerated; no new layout was introduced.

Goal, initial position and initial heading came from the full saved transport trial JSON for each layout/seed. The required target heading is the wrapped sum of initial yaw and initial heading error. The paired control experiment already verified the shared initial goal and initial-pose tolerance; this probe's route starts at the exact transport initial position. The JSON records each source trial path and CSV, summary and config SHA256 values, mesh hashes, source hashes, receipt hash and the reproducible CLI hash. The initial throwaway probe SHA256 was `ba37965f38aa634dfd6182dd5e2d693ee65f43c8522bfa1671936057127dea5f`; the committed CLI hash is recorded as `probe_sha256` in the new report. It includes every route vertex.

The rock XY polygons are the exact rotated rectangular faces of the source-controlled cuboid meshes, including their full footprint. A point-to-polygon distance routine was checked with known points inside, beside and diagonally outside a unit square.

## Target clearance

The planner's square footprint has half width 0.7 m. A circle with radius `sqrt(2)*0.7 = 0.989949 m` contains this square at every heading. Every target-to-rock distance exceeds this radius plus 0.18 m. Thus the entire strict native goal disk has geometric rock clearance for the square at every heading, including the required heading interval. This statement concerns mesh XY geometry; it does not by itself establish scan-mask acceptance.

| Layout / seed | Minimum goal-to-rock distance m | Required target heading rad | Entire strict goal disk clear for square at all headings |
| --- | ---: | ---: | --- |
| 604 / 103 | 1.215612 | -0.446646 | Yes |
| 611 / 110 | 3.397800 | 0.857908 | Yes |
| 614 / 113 | 5.878462 | -0.706482 | Yes |
| 615 / 114 | 1.579725 | 0.520614 | Yes |

A synthetic 101 by 101, 5 cm scan was ray projected at each exact goal center from the deterministic ground triangles and rock top faces. The ground height uses the existing mesh's two-triangle cell interpolation; rock heights use the exact top vertices. Common sensor-height offsets can be omitted because `scan_costmap` uses height differences and median leveling. Existing `scan_costmap` was called without changing thresholds, inflation, padding or scan size.

At all four goal centers the hard-map center cost is 0.0 at the required heading and at offsets -0.099 and +0.099 rad. These three tested headings are strictly inside the native heading tolerance. The exact target center gives zero position error. This supplies an admitted goal pose in the synthetic scan model. It does not prove every disk point or every heading in the interval passes the scan classifier.

## Complete route certificates

The global search uses a 10 cm grid over [-8,38] m in both axes, inside the actual ground bounds [-25,55] m. It marks centers free only when their exact distance to every rock polygon exceeds `sqrt(2)*(0.7+0.05+0.05) + sqrt(2)*0.1/2 = 1.202082 m`. The first term is a 1.131371 m all-heading circumscribed clearance radius. It contains the 0.7 m square plus extra 5 cm edge and 5 cm sampling/rounding allowances. The second term covers half the longest 8-neighbor grid segment. Diagonal corner cutting is forbidden.

Each returned path connects the exact initial position to its first grid vertex, then follows grid segments to a point strictly within 0.18 m of the goal. A second distance check includes the exact initial position and every vertex. Distance to a closed polygon is 1-Lipschitz: the minimum vertex distance minus half the longest segment is a lower bound on rock distance everywhere along the complete polyline. Every returned path passes this bound with margin above 1.131371 m. Therefore these are continuous geometric clearance certificates within this planar model, not merely disconnected grid-cell examples. The final endpoint also passes the unchanged synthetic scan at its required target heading with center cost 0.0.

| Layout / seed | Complete route length m | Endpoint position error m | Continuous rock-distance lower bound m | Endpoint required-heading scan cost |
| --- | ---: | ---: | ---: | ---: |
| 604 / 103 | 12.079263 | 0.132677 | 1.132101 | 0.0 |
| 611 / 110 | 10.220298 | 0.140812 | 1.184985 | 0.0 |
| 614 / 113 | 10.359622 | 0.125779 | 1.159930 | 0.0 |
| 615 / 114 | 10.519331 | 0.119183 | 1.164189 | 0.0 |

The 604 route has the smallest certificate margin: approximately 0.000730 m above the deliberately enlarged 1.131371 m radius. This is a computed geometric bound against the supplied float32 mesh vertices. It is not a physical tolerance or tracking margin. Its clearance above the actual square's 0.989949 m circumscribed radius is approximately 0.142151 m.

## Command

Run from the repository root. Replace `/path/to/confirmation-raw` with the exported evidence root:

```bash
rtk proxy uv run --no-project --python 3.12 --with torch==2.10.0 --with torchvision==0.25.0 --with numpy==2.3.1 --with h5py==3.15.1 --with einops==0.8.1 --with pyyaml --with gdown==6.5.1 python -m scripts.diagnose_rover_clearance /path/to/confirmation-raw/fresh-test-rocks-{604,611,614,615}/*/seed-*-transport.csv --output global-clearance.json
```

The command asserts input mesh hashes, elementary geometry checks, strict endpoint goal distance and the continuous-segment clearance bound. Like `diagnose_rover_timeouts`, the CLI accepts CSV paths and optional `--output`. It loads each matching trial JSON and folder config. It defaults `--mesh-receipts` to this repository’s committed mesh receipts. Without `--output` it writes the full JSON report to stdout. It needs no GPU, network request, training or simulation job.

The committed CLI was run on all four saved transport traces. All assertions passed, and the mesh hashes, target scan results, complete path vertices and certificate values matched the original external probe exactly. The report's `probe_sha256` matches the committed CLI source. No extra test framework was added.

## Limits and next step

This probe rules out geometric target/route impossibility from the rocks for 604, 611, 614 and 615 under the stated planar footprint model. The existence result does not depend on failure to find a route on a finite grid: each returned polyline has a continuous distance certificate. A different case with no grid route would remain inconclusive without a separate barrier argument.

The route certificate is orientation independent for the geometric square, so arbitrary heading at a path vertex and final alignment fit the clearance model. It does not model steering limits, inertia, traction, body tilt, wheel articulation, body-rock contact, native contact detection, or tracking error. The extra edge/raster allowances are geometric margins, not a proof of the complete scan classifier at every possible path pose. Only the reported goal-center and endpoint synthetic scans were scored. Actual ray timing, observed coverage, forecast history and control decisions still require the native HDF replay. Saved final-scan local reachability alone never established global reachability; this separate full-mesh construction supplies the bounded global result.
