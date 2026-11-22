The combined waypoint controller exposed a heading failure during arrival
acknowledgement on layout 501, seed 30. Its existing centering branch scored only
position. At step 304, distance and heading were both inside their strict bounds.
The next command moved inward but crossed the heading bound.

The public planner regression check failed at 0.101741201 rad before the change.
The captured native value is 0.101741314 rad. The test uses the measured motion
plus the frozen model's change for another command. It is diagnostic evidence,
not proof of native physics.

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

The correction scores the next acknowledgement step by the larger of predicted
distance / 0.18 m and absolute heading / 0.1 rad. It applies only while the
existing centering branch is active. The full rollout still checks measured-scan
risk. Candidate commands, rollout lengths, safety rejection, forecast weights,
waypoint logic, arrival bounds, and the native verifier are unchanged.

The captured pre-step-305 state now selects reverse -0.2 and turn +0.23. Its
20-step measured-scan risk is 0.05, below the unchanged hard rejection bound 100.
The focused planner suite passes all 13 tests, including the earlier layout 613
arrival recoil and original layout 501 zero-command arrival cases.

This is a provisional controller slice. Native layout 501/30 and 613/112 runs
must pass before the full known-case gate is repeated. The failed `649eb0b`
selection gate stays failed. No fresh-layout comparison is authorized by this
local result.
