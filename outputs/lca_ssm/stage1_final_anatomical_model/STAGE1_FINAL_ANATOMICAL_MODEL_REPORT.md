# Stage‑1 Final Anatomical Model Report

Generated: 2026-08-07T15:15:18.745370+00:00

## Outcome

- Accepted source cases: **133**
- Existing assignments retained: **14**
- Daughter roles resolved by swapping: **0**
- Assignment unresolved/manual review: **119**
- Insufficient RCA reference geometry: **0**
- LCX crown supported: **14**
- LCX crown ambiguous: **0**
- LCX crown not supported: **0**
- Crown not evaluated because assignment/reference remained unresolved: **119**
- Final clean anatomical-model cohort: **14**
- Primary representative: **189.label**
- Secondary representatives: **37.label, 192.label, 148.label, 20.label, 133.label, 102.label, 89.label, 190.label**

## `118.label` diagnostic

- Previous saved assignment: **LAD=branch_a, LCX=branch_b**
- New result: **unresolved_manual_review**
- Assignment 1 / 2 scores: **-0.08920192260054313 / 0.08920192260054313**
- Score margin: **0.17840384520108626**
- Winning evidence votes: **5/8**
- LCX crown result: **not_evaluated_unresolved_assignment**

## Scientific interpretation

The model is dataset-derived. Source centrelines were not anatomically corrected. Cases inconsistent with the intended anatomical abstraction were quarantined rather than geometrically forced to conform. The inferred RCA component is not annotated RCA ground truth. This is a pre-generative anatomical/statistical preparation stage, not a clinical coronary model.

## Branch-role resolution

Saved LAD/LCX arrays were reconstructed as neutral `branch_a` and `branch_b` from their saved provenance. Both assignments were scored with pairwise, within-patient normalized contrasts. Four LAD measurements (inferior reach, terminal inferior displacement, downward dominance, and progression to the inferior point) form one averaged evidence group. Four LCX measurements (RCA-plane residual, in-plane travel, in-plane tangent behaviour, and angular coherence) form a second averaged group. Repeated measurements are averaged within their role group before the two groups are combined. Automatic resolution requires at least 5/8 independent votes, score margin ≥ 0.16, winner score ≥ 0.08, and a minimally coherent proposed LCX. Otherwise the case remains unresolved.

## Independent LCX crown validation

The independent plane is fitted from all unchanged inferred-RCA candidate points only. LCX never contributes to this validation plane. Candidate intervals are continuous source-index ranges beginning within the proximal 20% and spanning at least 25%. No ellipse, smoothing, isolated-point rejection, or fitted curve participates. Crown classes use explicit multi-signal geometric QC criteria—not cohort medians and not clinical thresholds—including absolute and normalized plane residuals, in-plane path/tangent fractions, angular sweep, monotonicity, and coverage. A near-full branch with poor plane behaviour cannot be promoted to supported.

## Plane fitting mathematics

For relevant unchanged points `P`, `C = mean(P)`, `Q = P - C`, and `U, S, Vᵀ = SVD(Q)`. The plane normal is `n = Vᵀ[-1]`, giving `n · (x - C) = 0`. The final coronary measurement plane uses inferred RCA plus the independently validated continuous LCX crown interval. The LAD plane uses every unchanged resolved-LAD point. Plane separation is `acos(|n_cor · n_lad|)`. Display planes are parallel copies translated to the LMCA bifurcation and are never used for residual measurement.

## Dataset-derived ellipse references

All ellipse landmarks are exact source points selected at normalized arc-length positions. The coronary ellipse uses sparse landmarks from the inferred RCA and validated LCX interval. The LAD affine ellipse uses sparse ordered LAD landmarks and exact endpoints. Solid arcs indicate source-supported parameter intervals; dashed arcs are extrapolated full references. Landmark and full-source descriptive residuals are reported, but ellipse residuals neither validate LCX nor modify vessels.

## Canonical frame

One right-handed rigid frame is constructed per case from the final measured plane normals. Canonical X follows the plane intersection/crown direction, X sign follows resolved LCX progression, and Z is the signed coronary-plane normal with LAD descent toward negative Z. Every branch receives the same `p_canonical = R @ (p_original - B)` transform. No scaling or branch-specific transform is used. Determinants, orthonormality, and segment-length preservation are validated.

## Source-integrity proof

Every source file and array is reloaded after analysis. Point counts, coordinate hashes, coordinates, and original segment lengths must match exactly. Display-transform segment lengths must agree within floating-point tolerance. The final validation records the maxima across the cohort.

## Limitations

- RAS inferior direction and inferred RCA geometry provide engineering evidence, not annotated coronary labels.
- The RCA component is inferred from disconnected source topology and may not represent true RCA anatomy.
- Explicit geometric QC limits are transparent engineering choices, not clinically validated thresholds.
- Ellipses are simplified descriptive references and do not reproduce every noisy centerline point.
- Only clean cases support this two-plane abstraction; excluded cases remain available in QC tables.
- No PCA or generative geometry is used or trained in this increment.
