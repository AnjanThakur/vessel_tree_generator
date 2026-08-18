# Real-versus-generated cohort validation

This directory compares the accepted synthetic cohort with resolved, anatomy-gated immutable real-patient references. KS p-values are descriptive only because the generated cohort is intentionally smaller and individual samples are produced by empirical bootstrap plus low-scale PCA innovation.

Status: `PASS`. Compared metrics: 41; descriptive passes: 17; warnings: 24.

RCA decision: excluded from the primary cohort because the available RCA paths are inferred disconnected candidates rather than resolved ground truth; optional RCA support remains in the code and baseline audit.

## Descriptive warnings

- `bifurcation_angle_deg`
- `ellipsoid_a_mm`
- `ellipsoid_b_mm`
- `ellipsoid_c_mm`
- `lad_length_mm`
- `lad_obliquity_rad`
- `landmark_bifurcation_offset_mm`
- `landmark_bifurcation_u_rad`
- `landmark_bifurcation_v_rad`
- `landmark_lad_endpoint_offset_mm`
- `landmark_lad_endpoint_u_rad`
- `landmark_lca_ostium_offset_mm`
- `landmark_lca_ostium_v_rad`
- `landmark_lcx_endpoint_offset_mm`
- `landmark_lcx_endpoint_u_rad`
- `landmark_lcx_endpoint_v_rad`
- `lcx_length_mm`
- `lcx_obliquity_rad`
- `lcx_tortuosity`
- `lmca_length_mm`
- `lmca_obliquity_rad`
- `lmca_tortuosity`
- `pca_mode_01_standard_score`
- `pca_mode_04_standard_score`
