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

`arrival-heading-501-30-joint.npz` captures the subsequent native probe of the
rejected `3d853af` controller. Its SHA256 is
`9e454ceb7351c9f7c8a8cabb3d594dd62f9c4df8af791433b5c685720d77ab73`.
It has the same pre-command state. Reverse -0.2 and turn +0.23 reaches distance
0.148812359 m and heading 0.118715525 rad. The action improves distance and
worsens heading. A frozen-model differential with a constant measured residual
had predicted heading would improve. Native physics disproved that estimate.

The public `plan_action` test checks that both captured heading-limited arrivals
select zero speed and zero turn. It keeps the actual commands and native heading
errors explicit. It makes no claim about the unmeasured zero-command endpoint.
The frozen model weights come from `arrival-613-112.npz`. Native runs must confirm
the brake result.
