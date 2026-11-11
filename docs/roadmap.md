# Roadmap

Build the first rover world model in three stages. Complete the two independent
acceptance checks in [the requirements](../REQUIREMENTS.md) before connecting
the rover to JEPA-WMs.

## 1. Complete the reference evaluation

Resolve the upstream Metaworld dependency failure described in
[results](results.md#reference-planning). Then run the existing Step 1 job.

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

Accept this stage when sample sequences load with aligned transitions and never
cross episode resets. The current position-only trace lacks the observations
and action labels required for this dataset.

## 3. Train and evaluate a small baseline

Reuse JEPA-WMs with a rover dataset adapter and an encoder suited to the recorded
observations. Fit a tiny batch first to check the data and training path. Then
train on the starter dataset.

Measure one-step and multi-step prediction on held-out terrain. Compare physical
state predictions with a baseline that holds the current state constant. Check
that changing an action changes the predicted motion. Save checkpoints and the
evaluation configuration.

After the predictive baseline works, add a planner and measure goal progress,
collisions, and terrain-dependent failures. Add an interpretable physical-state
prediction head before physics losses. Learned feature dimensions must not be
treated as physical coordinates. Compare physics losses with the same baseline
and allow real wheel slip.
