The first-step model correction in `3d853af` failed native validation. The next
candidate restores the previous centering cost and brakes when heading has less
normalized margin than distance. Native validation of this brake is pending.

The combined waypoint controller exposed a heading failure during arrival
acknowledgement on layout 501, seed 30. Its existing centering branch scored only
position. At step 304, distance and heading were both inside their strict bounds.
The next command moved inward but crossed the heading bound.

The previous public planner regression check failed at 0.101741201 rad before
the first correction. The captured native value is 0.101741314 rad. That check
used the measured motion plus the frozen model's change for another command.
Native physics later disproved that estimate.

Three causes were checked in order:

1. Position-only centering ignores the heading margin. Adding joint normalized
   distance and heading at the rollout endpoint still chooses the failing
   reverse -0.05, turn +0.04 command. Mean and worst joint cost over the full
   rollout also keep that command.
2. Reverse motion and turn response matter. The captured +0.04 command increases
   heading error. Stronger positive turn reduces it in the frozen model's
   differential response. There is no evidence for inverting the turn command
   merely because speed is negative.
3. The five-field motion model misses the yaw motion at this state. It predicts
   0.093800903 rad after the recorded command; the observed value is
   0.101741314 rad. The prior command is not measured velocity. No model weights
   or state fields were changed.

The rejected provisional correction scores the next acknowledgement step by the larger of predicted
distance / 0.18 m and absolute heading / 0.1 rad. It applies only while the
existing centering branch is active. The full rollout still checks measured-scan
risk. Candidate commands, rollout lengths, safety rejection, forecast weights,
waypoint logic, arrival bounds, and the native verifier are unchanged.

The rejected candidate selected reverse -0.2 and turn +0.23 for the captured
pre-step-305 state. Its 20-step measured-scan risk was 0.05, below the unchanged
hard rejection bound 100. Its focused planner suite passed all 13 tests,
including the earlier layout 613 arrival recoil and original layout 501
zero-command arrival cases.

Native `3d853af` results reject that candidate. Layout 501/30 model-hold reaches
distance 0.148812359 m and heading 0.118715525 rad after the stronger reverse
command. Its paired no-hold control reproduces heading 0.101741314 rad.
Both have native acknowledgement and fail the strict heading rule. Layout
613/112 transport and JEPA pass, with zero collisions across all four trials.
The constant-residual estimate in the first unit test did not predict the new
action's physics. It has been replaced by a test that checks the brake command
and records both actual failing actions and native endpoints.

Pinned RLRoverLab source `578906ad5c3383087aeec038be82b87e05d4ee2c` explains
the sign risk in `rover_envs/mdp/actions/ackermann_actions.py`. In the Ackermann
branch, steering angles depend on turn direction. Wheel speeds are multiplied
by the sign of linear speed. Reversing speed keeps the steering angles and
reverses wheel speeds. The point-turn branch has different behavior. Both
failing commands use the Ackermann branch: radii 1.25 m and 0.869565 m exceed
the point-turn bound 0.7152 m. The source SHA256 is
`ca74532b5fd0fac5ef8f536c98a26a3c30e054765486f5ca8d192e7650d45f88`.
Evaluating the source function gives front-left/front-right wheel targets
-0.173/-0.327 rad/s for reverse -0.05, turn +0.04, and -0.55725/-1.44275 rad/s
for reverse -0.2, turn +0.23. Zero speed and turn set all wheel and steering
targets to zero. These target values explain the direction risk; they do not
prove the resulting body motion on terrain.

The next candidate centers only when current distance / 0.18 m exceeds absolute
heading / 0.1 rad. It uses the existing reached-goal and translation checks.
Otherwise it uses the existing zero-command hold. In the captured 501 state,
the fractions are 0.957106 and 0.975993, so it brakes. The earlier 613 recoil
fixture still centers because its position is closer to its bound. No new
threshold, model cost, horizon, safety mask, or weight is introduced.

Native layout 501/30 and 613/112 runs must pass before the full known-case gate
is repeated. Both failed candidates stay preserved as development evidence.
No fresh-layout comparison is authorized by these local results.
