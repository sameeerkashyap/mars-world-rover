# Pilot review and corrections

Reviewed implementation: [pilot PR](https://github.com/sameeerkashyap/mars-world-rover/pull/10),
initial commit `0caa1eb`. Independent Standards and Spec reviews followed the
Matt Pocock code-review skill. Corrections are in `d162981` and `fc0d29d`.

## Standards

One non-blocking finding: the full-feature pilot computed PCA and discarded its
result. This wasted CPU work. The pilot now uses `project=False`, which skips
covariance, eigendecomposition, and latent projection. The shared PCA interface
remains for the existing scaled trainer. The recheck accepted this correction.

No material import-placement, inline-comment, or Nautilus time-limit violation
was found.

## Spec

Four partial requirements were found and corrected:

1. Rotating nearby windows did not ensure different action-donor episodes.
   Donors now come from a different episode, with the mapping saved.
2. Global errors hid local failures. Reports now include episode groups,
   recorded terrain tags, and errors on the central 2-by-2 patches.
3. The existing residual predictor was missing. It and its own motion-transport
   control now run on the same windows. Reconstruction preserves feature
   components outside its PCA subspace.
4. Selected weights had no recorded digest. Reports now save checkpoint hashes.

The recheck confirmed all four corrections. Both reviewers also found an
input-order edge case: set-based digest checks accepted reversed validation
recordings, which changed positional window indices. Ordered digest checks
and a regression assertion now reject this case.

The candidate gate also requires improvement over copying and the current
residual predictor at both horizons. A separate evaluator scores existing
frozen weights without changing training or opening test recordings.

## Checks and limits

The isolated PR branch passed 36 relevant tests. The revised training command
completed a one-epoch real-recording check for both architectures. Its separate
evaluator also completed on the earlier local preflight weights. These checks
establish execution and data alignment, not model or navigation quality.

The current validation data covers one debug layout. Terrain-tag grouping
does not distinguish future procedural layouts with the same tag; episode
identity remains available. Such data needs unique layout metadata before a
new generalization study. Running v1 jobs keep their original source and are
left to finish; their selected weights require revision-2 evaluation.

Original findings: Standards 1 (unused PCA), Spec 4 (including weak action
donors). The input-order follow-up is fixed. No unresolved blocker was found
for the configured single-recording pilot.
