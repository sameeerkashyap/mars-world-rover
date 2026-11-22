# Measured waypoint retention candidate

Ticket: https://github.com/sameeerkashyap/mars-world-rover/issues/21

The planner can reuse one previous local waypoint. It must be finite, inside the
current scan, free at its exact 0.05 m hard-mask sample, and reachable in the
existing 0.1 m Dijkstra graph. Invalid or disconnected points use a new route.
Every predicted command still passes the current measured-scan safety check.

The release distance is 0.6 m, half the existing 1.2 m lookahead. Retention stops
within 1.2 m of the true goal and in terminal or alignment control. These values
were fixed before native evaluation. Graph edges, masks, candidate commands,
horizons, costs, forecast weight, and learned weights are unchanged.

The native runner owns one world point per trial. It transforms this point with
the current measured pose, updates it from the actual selected waypoint, and
clears it after stops or precision control. Decision recording does not change
control. Forecast audit receives the same pre-call hint and cannot update memory.
Original planners omit the unsupported input.

The compact `tests/fixtures/waypoint-611-110.npz` contains two adjacent native
observations. The public planner replay matched the recorded turn reversal.
Before the fix, the regression assertion failed with turn-sign product
`-0.1224999949336052`. With the valid retained point, the assertion passes and
candidate count, safe count, and measured rejection rules remain unchanged.

Focused checks cover valid, blocked, disconnected, outside, non-finite,
malformed, reached, near-goal, alignment, arrival, and unsafe-stop inputs. Runner
checks cover current-pose conversion, per-trial reset, stop clearing, audit hint
and counter preservation, legacy calls, and recording parity.

The [native development replay](waypoint-retention-native-validation.md) completed
on layouts 604/611/614/615, 613/112, 601/100 and 501/30-32. The fixed gate failed
because corrected 501/30 lost an original strict success at the heading boundary.
The candidate is not approved for fresh evaluation. The captured action replay
alone does not prove closed-loop progress, physical safety, forecast benefit or
performance on new layouts.
