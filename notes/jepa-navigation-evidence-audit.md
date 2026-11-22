# Frozen v12 JEPA navigation evidence audit

Date: 2026-11-21. This is a source and saved-report audit. No new simulator run or cluster job was started for this note.

The existing evidence does not establish a navigation benefit from the frozen v12 learned terrain forecast over its matching motion-only transport. On the six previously fresh trials, both controllers reached all six goals with no collisions. JEPA shortened the mean path by 0.065849 m, or 0.72% of the transport mean, and increased mean planning latency by 22.7%. Those trials cover only two rock layouts. The later predictor pilot does not supply new navigation evidence. Its new models failed their offline gate and never controlled the rover. [Saved navigation outcomes](rover-navigation-v2-verification.json), [pilot report](world-model-pilot-results.md).

## What the prior paired results show

The required control is `transport`, which uses the same motion prior, latent representation, terrain readout, candidate commands, measured-scan checks, and scoring weight as `jepa`. The extra learned residual is the change being tested. `model` is useful context but does not isolate that change because it has no terrain forecast cost. [Comparison design](../deploy/nautilus-jepa-navigation/README.md), [forecast implementation](../src/rover_forecast.py), [JEPA and transport motion](../scripts/train_rover_jepa.py).

| Phase | Matching transport | JEPA | Paired interpretation |
| --- | --- | --- | --- |
| Development, six cases | 3 successes, 0 collisions | 4 successes, 0 collisions | One JEPA-only success; no reverse discordance. These cases were inspected during development. |
| Previously fresh, six cases | 6 successes, 0 collisions | 6 successes, 0 collisions | No success or collision gain. Four JEPA paths were shorter; two were longer. |

Fresh paired path differences, JEPA minus transport, are -0.008958, +0.005009, -0.021545, -0.119655, +0.010175, and -0.260120 m for seeds 30–35. Their mean is -0.065849 m. Mean per-trial planning time is 0.067511 s for JEPA and 0.055010 s for transport. These arithmetic checks use the saved JSON, not a new trace verification. Most path gain comes from seeds 33 and 35, both on layout 502. Three trial seeds on one mesh do not create three independent terrain samples. [Saved per-trial results](rover-navigation-v2-verification.json), [fixed cases and meshes](../deploy/nautilus-jepa-navigation/test-plan.json).

The development mean path over all trials is not an efficiency comparison: some trials stop without reaching the goal. Restricting that phase to its three common successes makes JEPA paths 0.051582 m longer on average. The six fresh trials are all common successes, so their path comparison has no such completion imbalance. [Saved outcomes](rover-navigation-v2-verification.json).

The scan-only seed-30 failure is a separate terminal-control issue. Goal hold later changed that result on inspected terrain. It is not evidence that JEPA supplied better terrain information. The next comparison should use the same goal-hold behavior in both arms. [Goal timing diagnostic](rover-goal-timing-diagnostic.json), [goal-hold report](world-model-pilot-results.md), [current runner](../deploy/nautilus-navigation/rover_navigate.py).

## Can this objective change a command?

Yes. After measured-scan rejection, the planner adds `0.25 * predicted_cost` to each safe candidate's waypoint cost. The returned action is the first command of the lowest-cost candidate. The readout returns values in [0, 1], so each candidate's extra penalty lies in [0, 0.25]. A base-cost gap above 0.25 cannot be reversed by this penalty alone. A change in the winning candidate also need not change the executed command: several turn-then-drive candidates share their first turn command. The existing unit test shows that a synthetic terrain cost can change an action; it does not show that frozen v12 costs do so on real observations. [Planner](../scripts/rover_plan.py), [readout](../src/terrain_cost.py), [reranking test](../tests/test_rover_forecast.py).

The measured scan already supplies the route and rejects unsafe or unknown candidates. Forecast scoring can only rerank the survivors. It cannot open a route rejected by the scan. It is disabled during final heading alignment and while the one-second history fills. Goal hold can return before candidate scoring. The readout predicts a geometric blocked-area fraction learned from scan height rules; it does not predict traction, slip, physical stability, or calibrated collision probability. These constraints reduce the kinds of navigation gain this experiment can reveal. [Planner](../scripts/rover_plan.py), [history and forecast](../src/rover_forecast.py), [terrain labels](../src/terrain_cost.py).

The forecast horizon does not cover every candidate's motion horizon:

| Candidate | Motion scoring horizon | Learned forecast |
| --- | --- | --- |
| Constant command | 20 ticks = 4 s | First 4 s |
| Turn 10 ticks, then drive 20 ticks | 30 ticks = 6 s | First 5 s: 2 s turn + 3 s drive |
| Turn 20 ticks, then drive 20 ticks | 40 ticks = 8 s | First 5 s: 4 s turn + 1 s drive |
| Turn 40 ticks, then drive 20 ticks | 60 ticks = 12 s | First 5 s of turn only |
| Turn 60 ticks, then drive 20 ticks | 80 ticks = 16 s | First 5 s of turn only |

The measured-scan check still covers the full motion horizon. The learned penalty uses the maximum readout of the central four patches at complete one-second forecast intervals. Thus long turns can win based on a future driving endpoint that the learned terrain score never evaluates. This is a declared limit of the current objective, not grounds to change it after fresh results are inspected. The experiment tests this existing integration; a horizon redesign would require a separate development and fresh test. [Candidate construction](../scripts/rover_plan.py), [forecast clipping and active intervals](../src/rover_forecast.py).

## Small fresh protocol

The proposed fixed experiment is 20 new independent rock layouts, seeds 601–620, with one trial seed 100–119 per layout and two matched arms: transport and frozen v12 JEPA. This is 40 trials. Use the original rock generator parameters, same goal-motion model, same terrain readout, same weights, same scoring weight, and same candidate set. Enable goal hold in both arms. Balance controller order by trial-seed parity. Two shards of ten layouts are acceptable if both run the same frozen code and simulator configuration. Do not retrain or tune from these outcomes. [Original frozen plan](../deploy/nautilus-jepa-navigation/test-plan.json), [current generator](../src/evaluation_terrain.py), [current runner](../deploy/nautilus-navigation/rover_navigate.py).

The current generator accepts only seeds 501 and 502. A fresh runner therefore needs a new frozen plan and the smallest generator change that admits exactly the new fixed cases. Current planner and runner hashes also differ from the historical plan because goal hold was added. Preserve the historical source snapshots and plan. Hash the new source, weights, dependency versions, simulator task configuration, generated meshes, case list, controller order, and analysis rules before opening fresh outcomes. [Generator seed guard](../src/evaluation_terrain.py), [historical reproduction warning](../deploy/nautilus-jepa-navigation/README.md), [hash checks](../deploy/nautilus-navigation/rover_navigate.py).

For each pair, reset the same simulator seed and terrain, settle in the same way, and check the measured initial pose and goal. Also save and compare initial scan, linear/angular velocity, joint state, and target heading. The old verifier checks position, yaw, and goal but does not establish equality of all these other states. Keep strict native success plus terminal distance <0.18 m, heading error <0.1 rad, and no collision. Preserve terminal measurements before automatic reset. [Runner reset and snapshot](../deploy/nautilus-navigation/rover_navigate.py), [verifier](../deploy/nautilus-jepa-navigation/verify.py).

For pinned IsaacLab commit `ffff603eafc6b74264a5261cc0183d6a65390d78`, the PhysX data fields are `robot.data.root_link_vel_w.torch[0]` for six world-frame linear/angular velocity values, and `robot.data.joint_pos.torch[0]` / `robot.data.joint_vel.torch[0]` for joint state. Separate `root_link_lin_vel_w` and `root_link_ang_vel_w` fields also exist. These getters return the same proxy-array type used by the runner's existing position getter. [Pinned official root velocity implementation](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py#L633), [joint getters](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py#L936).

Use the existing fixed 750 control-step evaluation horizon in both arms. This is an episode step count, not a wall-clock limit on a Nautilus job. Complete the fixed case set without interim significance checks or outcome-based case replacement. An invalid initial state or infrastructure error must be reported, with its retry recorded. Do not silently drop difficult cases. Do not add a Job deadline, timeout wrapper, or automatic timed deletion.

Record a compact mechanism check during the same rollouts. On the exact same current history, evaluate both forecast modes before taking the selected arm's action. Record their chosen first commands, candidate-cost ranges, winning margins, ready steps, and whether scoring was skipped for alignment, hold, or no safe path. Cross-arm trajectories diverge after different actions; comparing commands at equal elapsed steps then does not isolate forecast influence. Existing scoring-call counts show that the hook ran, not that the command changed. A shared-history comparison solves that narrow evidence gap. Its extra diagnostic work must be outside the selected arm's measured planning latency. [Current counters and timing](../src/rover_forecast.py), [runner timing](../deploy/nautilus-navigation/rover_navigate.py).

## Decision rules to freeze

The primary endpoint is strict verified success. Let W be JEPA-only successes and L be transport-only successes. Report all four paired outcomes, success difference `(W-L)/20`, and the exact two-sided binomial test of W successes among W+L discordant pairs under probability 0.5. With no discordant pairs, record no measured success difference and use p=1 by an explicit analysis convention. Do not count control steps or forecast calls as independent samples. Each distinct layout supplies one pair. The exact binomial API and confidence interval implementation are documented by [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html).

Use the proposed promotion rule: success gain at least 0.10, exact two-sided p <=0.05, no extra collision hazard, and mean JEPA planning latency <=0.2 s. The collision condition should check individual matched cases as well as total counts: reject promotion if JEPA collides on any case where transport does not. Save per-trial p95 and worst planning latency too. A mean below 0.2 s alone does not show that every command meets a 0.2-second control interval. Success and collision are simulator outcomes, not a physical safety claim.

The sample has limited power at the prior observed success ceiling. Even a 2/20 JEPA success gain with no losses has exact two-sided p=0.5. The smallest all-favorable discordance that passes p<=0.05 is six wins and zero losses: `2 / 2**6 = 0.03125`. This is a 30-point observed success gain. These values follow directly from the stated binomial model; the 10-point practical threshold does not make a 10-point gain detectable with this small sample. Do not claim the planned sample is powered to detect that margin.

Use these outcome statements:

- **Benefit demonstrated for this fixed integration and layout family:** every promotion condition passes, pairing and source checks pass, and the selected actions have a verified route from forecast costs to control. This does not establish gains on other terrain classes.
- **No benefit observed:** the valid fixed sample has no higher JEPA success count, or JEPA has a clear collision or latency regression. Report the measured differences. Do not equate this phrase with proof that the true benefit is zero.
- **Insufficient evidence to establish a benefit:** a positive success estimate fails the promotion rule; the sample reaches a ceiling with mostly tied successes; the objective rarely changes commands; or missing, unpaired, changed-source, or incomplete cases prevent the planned comparison. A complete all-success tie can also be described as “no measured success gain,” while remaining insufficient to rule out benefit elsewhere.

Keep path length, simulation steps to success, stalls, final error, planning p95, and shared-history command disagreement descriptive. Compare path efficiency on common-success pairs and report how many pairs that excludes. Do not promote based on a favorable secondary metric after the primary endpoint fails. If path efficiency is the intended product benefit, define its practical margin and a fresh primary test in a new plan before gathering another sample.

## Audit limits

I inspected the local code, frozen plan, saved aggregate/per-trial outcomes, and goal-hold and predictor-pilot reports. I recomputed the paired path and latency arithmetic from the saved navigation JSON. I did not retrieve the historical CSV files, checkpoint tensors, original source snapshots, simulator task dependencies, or fresh results. The existing report was therefore not independently regenerated here. The current verifier also trusts summary forecast-call counts; those counts are not reconstructed from CSV. [Current verifier](../deploy/nautilus-jepa-navigation/verify.py).
