`arrival-heading-501-30.npz` is the exact last four pre-command decisions from
layout 501, seed 30, model-hold in the known-case comparison of source `649eb0b`.
The NPZ SHA256 is
`713d185078456bd38feac0fbb35772fbdf2aeba7cd25cfcc058e91d7abd0c88a`.
It contains policy states and measured scans, commands, poses, planner diagnostics,
the last CSV rows, the native summary, and source and extraction hashes.

The last command starts at distance 0.172279030 m and heading 0.097599268 rad.
It repeats reverse speed -0.05 and turn +0.04. The recorded endpoint has native
acknowledgement, distance 0.164792120 m, and heading 0.101741314 rad. That endpoint
fails the unchanged strict heading bound of 0.1 rad.

The public `plan_action` regression test replays the recorded command's measured
motion exactly. For another command it adds the frozen model's motion difference
from that command. The model weights come from `arrival-613-112.npz`.
This checks command selection against the captured failure. It does not prove
another command's native physics. Native runs must confirm the result.
