# Eligibility recovery audit

The audit retained the 52-case PCA cohort. No anatomical threshold was relaxed.

## Exclusive funnel outcomes

- `centerline_or_landmark_extraction_failed`: 9
- `core_anatomy_gate_failed`: 116
- `daughter_assignment_unresolved`: 10
- `ellipsoid_invalid`: 13
- `pca_eligible`: 52

## Findings

- All 191 extracted cases have valid derived cardiac frames; frame construction is not the limiting stage.
- Ten of 191 extracted cases remain below the predeclared multi-signal assignment confidence gate and are excluded.
- The main reduction is anatomical-role incompatibility. These include LAD/LCX course relationships that cannot safely be repaired numerically without changing source anatomy.
- Additional cases fail ellipsoid quality checks driven by underconstrained/long-tailed ellipse axes or center separation. The final sampler uses only positive, finite, jointly observed eligible axes.
- Surface reconstruction is numerically exact for the stored angular/radial representation. No projection implementation bug was found that safely recovers excluded anatomies.
- The 52 cases are independent eligible source anatomies. Synthetic phases, diseases and variants do not increase the independent training N.

## Decision

Retain N=52. Increasing N by lowering assignment/anatomy/ellipsoid gates would weaken the scientific contract. Future recovery requires expert-reviewed labels or a separately validated support model, not threshold manipulation.
