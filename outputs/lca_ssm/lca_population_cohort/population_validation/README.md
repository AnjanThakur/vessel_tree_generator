# Real-versus-generated cohort validation

This directory compares the accepted synthetic cohort with resolved, anatomy-gated immutable real-patient references. KS p-values are descriptive only because the generated cohort is intentionally smaller and individual samples are produced by empirical bootstrap plus low-scale PCA innovation.

Status: `PASS`. Compared metrics: 50; descriptive passes: 49; warnings: 1.

RCA decision: excluded from the primary cohort because the available RCA paths are inferred disconnected candidates rather than resolved ground truth.

## Descriptive warnings

- `lmca_max_resampled_turn_angle_deg`
