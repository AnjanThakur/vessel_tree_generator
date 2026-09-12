# Generator sampling-strategy audit

Three bounded strategies were evaluated without changing learned acceptance thresholds.

| Strategy | Accepted/candidates | Acceptance | Evaluation |
|---|---:|---:|---|
| A_current_empirical_bootstrap_plus_pca_0_04 | 52/64 | 81.2% | full dense generator with rejection sampling |
| B_pure_pca_scores | 4/52 | 7.7% | fixed-representation reconstruction plus dense B-spline and full validator |
| C_empirical_bootstrap_moderate_innovation | 46/52 | 88.5% | fixed-representation reconstruction plus dense B-spline and full validator |

## Decision

Retain Strategy A. It preserves exact case-matched empirical branch courses, adds a small joint PCA innovation, and passes the complete dense validator after transparent rejection sampling. Pure PCA-score sampling is useful as an experiment but loses residual/nonlinear course information and produces more anatomical failures. The moderate-innovation option adds diversity at a measurable validity cost. Acceptance alone was not the selection criterion; anatomical role, local turn, population support and novelty were considered together.
