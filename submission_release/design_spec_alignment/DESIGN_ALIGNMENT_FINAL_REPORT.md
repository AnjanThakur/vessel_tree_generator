# Design Alignment Final Report

## Verdict

The frozen release is an **ellipsoid surface-relative statistical generator**, not an ellipse-arc generator. The two ellipses measure scaffold dimensions; neither LAD nor LCX is snapped to a planar ellipse.

## Architecture findings

- Cardiac frame: 191/191 rigid-frame checks passed in the frozen evidence; LAD is oriented toward negative cardiac Z.
- Plane comparison: the measured LAD plane and simplified design-derived plane have median absolute normal difference 24.966 degrees (materially different). The measured plane is used downstream.
- Ellipsoid provenance: coronary ellipse supplies `a,b`; the cardiac-Z-aligned IV/LAD ellipse axis supplies `c`, with flagged stability handling for pathological partial-arc fits.
- Surface representation: every eligible source point has XYZ, normalized arc position, wrapped/unwrapped u, v, offset, scaffold XYZ and local tangent coefficients in the point-level trace. Maximum numerical reconstruction error is 3.640e-14 mm.
- PCA: the primary model is **surface-relative deviation PCA**, not raw XYZ PCA. It has 81 features (LMCA 5 + LAD 12 + LCX 10, each with tangent-u, tangent-v and normal coefficients), 52 independent anatomies, and 13 retained modes.
- Spline: the production baseline is a shape-preserving cubic B-spline through matched cardiac XYZ controls, followed by exact surface re-parameterization. This is partial rather than exact compliance with a pure u/v spline. A literal u/v candidate was evaluated on all 52 cases: 26/52 eligible LAD control paths reach a polar chart singularity, its P95 maximum path difference from the protected baseline is 5.374 mm, and its maximum absolute length change is 70.71%. It is therefore not promoted without a validated pole-safe multi-chart representation.
- Tortuosity/obliquity: population behaviour is inherited from the matched empirical surface trajectory and coordinated PCA. Extra independent perturbation is disabled to avoid double counting.

## Surface behaviour

| Quantity | Real mean | Generated mean |
|---|---:|---:|
| LAD v progression (rad) | 1.1114 | 1.1114 |
| LCX total u travel (rad) | 1.1467 | 1.1473 |
| LCX circumferential fraction | 0.6165 | 0.6166 |
| LAD apical fraction | 0.6599 | 0.6626 |
| LCX obliquity (rad) | 0.8875 | 0.8875 |
| LCX surface tortuosity (rad) | 0.1243 | 0.1115 |
| LCX absolute offset P95 (mm) | 22.4123 | 22.4972 |

- Current generated role-consistent count under the transparent surface rule: 40/52.
- Cause counts: `{"A_2D_PROJECTION_EFFECT": 2, "B_REAL_BASELINE_VARIATION": 12, "C_PCA_GENERATION_DRIFT": 0, "D_CARDIAC_FRAME_PROBLEM": 0, "E_BSPLINE_RECONSTRUCTION_PROBLEM": 0, "FULLY_DESIGN_CONSISTENT": 38, "F_VALIDATOR_TOO_WEAK": 0}`.

## Tree 0036 and 0052

- tree_0036: source 143.label; LCX circumferential fraction 0.749, LAD apical fraction 0.417; diagnosis `A_2D_PROJECTION_EFFECT`.
- tree_0052: source 199.label; LCX circumferential fraction 0.806, LAD apical fraction 0.259; diagnosis `A_2D_PROJECTION_EFFECT`.

## Dataset deviations

- RCA: `DELIBERATE_DATASET_DEVIATION`. Disconnected RCA candidates are not trusted annotated ground truth, so no RCA population model is claimed.
- Side branches: `NOT_IMPLEMENTED - DATA NOT SUFFICIENTLY VALIDATED`. Random diagonals, septals or OM branches are not fabricated.

## Integrity

- Source coordinate change: 0.0 mm.
- Source segment-length change: 0.0 mm.
- Protected source hashes unchanged: True.
- Frozen package hashes match manifest (9 files): True.

## Generator decision

No accepted cohort or frozen generator package was overwritten. The surface audit determines whether visual LCX concerns are projection/baseline effects or true drift. A pure u/v production spline is not promoted until a pole-safe chart is validated against the current acceptance and anatomy evidence.
