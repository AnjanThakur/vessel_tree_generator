# Generator method — future implementation boundary

The user requested that this increment stop immediately before the generative model. Therefore this document defines the hand-off and does not describe code that has been implemented.

## Available measured inputs

The future generator may consume the patient parameter table, circular statistics, continuous-variable correlation matrix, and pointwise residual table produced here. It must preserve cross-parameter dependence, the 180-degree axial nature of ellipse tilt, 360-degree landmark theta, and the empirical along-branch structure of residuals.

## Future steps (not implemented)

1. Sample the two ellipse parameter sets and their plane relationship from a joint population model.
2. Sample bifurcation and terminal angular positions using circular distributions and valid conditional extents.
3. Place ideal reference control points by normalized ellipse arc length.
4. Add smooth, seed-reproducible dataset-derived in-plane and plane-normal deviations while preserving endpoints and junction continuity.
5. Connect LMCA, LAD, and LCX at one bifurcation.
6. Reuse the repository's existing B-spline implementation rather than introducing another spline engine.
7. Validate topology, continuity, lengths, curvature, sampled support, residual magnitude, and real-versus-generated distributions.

No part of that sequence is executed by `ppt_priority_population_model.py`.
