# Roadmap

Build the first rover world model in three stages. Complete the two independent
acceptance checks in [the requirements](../REQUIREMENTS.md) before connecting
the rover to JEPA-WMs.

## 1. Complete the reference evaluation

Validate the existing Metaworld and dataset dependency overrides in a full
Step 1 run. See [results](results.md#reference-planning).

Accept this stage when checkpoint and preprocessor loading succeed, one dataset
sample passes inspection, and planning finishes with saved metrics. Preserve
the command, configuration, seed, hardware, source revision, and package versions.

## 2. Record a rover training dataset

Extend the existing recorder to save observations, applied actions, next
observations, time, episode boundaries, goals, and outcomes. Start with available
height maps and onboard state while camera capture remains unresolved.

Collect varied steering and speed commands across seeds and terrain. Split
whole terrain layouts between training, validation, and testing. Keep simulator
diagnostics separate from agent inputs.

The original recordings total 6,000 rows. The new collection completed 18,000
more rows across six seeds. Four new recordings are for training; Mars 20 and
debug 21 stay reserved for final evaluation. The recordings cover two existing
layouts. New layouts and independent motion labels are still needed for claims
about generalization.

## 3. Train and evaluate a small baseline

Reuse JEPA-WMs with a rover dataset adapter and an encoder suited to the recorded
observations. Fit a tiny batch first to check the data and training path. Then
train on the starter dataset.

The initial state/height-scan predictor lost to copying. The corrected
goal-motion model now beats copying at one and five steps on debug seeds 1
and 2. It handles circular angles and learns motion from commands. It does
not use terrain inputs. Reserve a fresh recording for final testing, then add
terrain observations and independent motion labels before claiming terrain
generalization. The adapted Meta DINO-WM predictor now trains on height scans;
it passed the fixed development and pooled reserved-test checks. Its final
feature error improved on copying by 10.80% at one second and 19.88% at five
seconds. Mars alone narrowly missed the one-second copy target. The next step
is to connect the frozen predictor to the planner and test goal completion,
collisions, and unseen layouts with the existing safety checks. See the
[training guide](../deploy/nautilus-jepa-rover/README.md).

Measure one-step and multi-step prediction on held-out terrain. Compare physical
state predictions with a baseline that holds the current state constant. Check
that changing an action changes the predicted motion. Save checkpoints and the
evaluation configuration.

Scan-aware routing now improves the original six cases from two to four
goals, with zero collisions instead of four. Six fresh cases also reached four
goals without collision, using frozen source and checkpoint. Next, improve conservative stops near narrow passages and
collect measured low-speed responses for precise goal tracking. Test more
terrain layouts and keep native collision and goal checks enabled. Add an interpretable physical-state
prediction head before physics losses. Learned feature dimensions must not be
treated as physical coordinates. Compare physics losses with the same baseline
and allow real wheel slip.
