# Final mathematical, anatomical, physiological and visual audit

This is an engineering and dataset-anatomical audit. It is not clinical validation.

## Finding register

| Area | Original finding | Classification | Final status / correction |
|---|---|---|---|
| Release protection | Current git state and protected roots needed a before-state | PASS | `PRE_FIX_HASHES.json` records git state and per-file hashes before corrections. |
| QC ellipsoid volume | Used a one-axis sine surrogate, omitted longitudinal shortening, sampled 0–0.9, and exceeded 100% | VISUALIZATION_ERROR | FIXED: true `100*(a/a0)*(b/b0)*(c/c0)` at explicit phases; stored minimum 66.645313%, continuous minimum 66.564000%, maximum 100%. |
| PCA curve in QC | Used a linear ramp to the final cumulative variance | VISUALIZATION_ERROR | FIXED: plots actual cumulative explained-variance ratios. |
| Cohort funnel | Began at 191 and omitted the 200-volume source inventory | DOCUMENTATION_ERROR | FIXED: 200 → 191 → 181 → 65 → 52. |
| Motion phase | Closure and phase values required cross-artifact verification | PASS | 56 audited rows; zero failures; phases are 0, 1/9, …, 1 with phase 1 repeating phase 0. |
| Accepted-tree role audit | Projection suggested possible LAD/LCX reversal in trees 0036 and 0052 | PASS_WITH_EXPLANATION | 52/52 pass the independently recomputed production coordinate gate; zero validator mismatch. 35/52 pass every stricter descriptive absolute-reach rule. |
| Tree 0036 | LAD looked circumferential in X-Z | PASS_WITH_EXPLANATION | Labels/frame/color are correct. Production role gate passes; descriptive failures: lad_has_minimum_inferior_reach;lad_has_longitudinal_course;lad_descent_is_sustained. |
| Tree 0052 | LAD looked short/circumferential in X-Z | PASS_WITH_EXPLANATION | Labels/frame/color are correct. Production role gate passes; descriptive failures: lad_has_minimum_inferior_reach;lad_has_longitudinal_course;lad_descent_is_sustained;lcx_has_substantial_absolute_lateral_reach. |
| Branch colors | Needed array-to-color identity evidence | PASS | Plot-series helper and regression test prove LAD coordinates are red and LCX coordinates teal. |
| 3D collision | 2D crossings could be misleading | PASS | Exact segment clearance and production nonlocal-clearance policies recomputed; collision IDs at 0.75 mm: none. |
| LMCA length | Mean/range are not comparable with typical clinical LMCA morphometry | DOCUMENTATION_ERROR | Endpoint rule traced. Repository segment is graph-root endpoint to selected major-daughter junction; report now calls it repository-defined LMCA/proximal-LCA path. No literature clamp applied. |
| Pulsatility timing | Pulse reused motion peak 0.35 despite charter and early-diastolic IVUS evidence | PHYSIOLOGICAL_MODEL_ISSUE | FIXED: motion peak remains 0.35; independent pulse peak is 0.60 (early diastole); literature disagreement is explicit. |
| Disease mathematics | Severity and minimum-lumen semantics needed independent calculation | PASS / DOCUMENTATION_ERROR | Exact radius/area reductions recomputed with zero XYZ change; ambiguous phrase replaced by `minimum lumen diameter (mm)`. |
| B-spline | Endpoint/topology/curvature needed independent evidence | PASS | Finite composite cubic B-splines; maximum endpoint and shared-bifurcation errors are zero; surface-coordinate reconstruction agrees to numerical tolerance. |
| Curvature | Total turn is sample-sensitive | PASS_WITH_EXPLANATION | Added 1 mm uniform-arc Menger curvature; real/generated P95/P99 are descriptive, not clinical limits. |
| Raw ellipse tails | Full ellipse axes could be mistaken for heart diameter | DOCUMENTATION_ERROR | FIXED caption: raw 191-case incomplete-arc reference fits, separate from final 52-case scaffold cohort. |
| PCA plot labels | Reference values were visually unexplained | VISUALIZATION_ERROR | FIXED titles and expected references: SD ratio=1 and generated mean=0. |
| Report anatomy claim | `every accepted tree` wording implied all absolute criteria | DOCUMENTATION_ERROR | FIXED: every tree passes the documented relative role gate; absolute apex-reach checks are separately descriptive. |

## Numerical anchors

- Repository LMCA/proximal-LCA path: N=52, mean 17.176605 mm, SD 15.062155 mm, range 4.472325–72.779974 mm.
- Ten longest source cases: 150.label, 5.label, 185.label, 194.label, 155.label, 113.label, 72.label, 7.label, 134.label, 177.label.
- Generated LAD curvature: P95 0.146905 mm⁻¹, robust maximum/P99 0.269943 mm⁻¹.
- Real eligible LAD curvature: P95 0.142940 mm⁻¹, robust maximum/P99 0.270163 mm⁻¹.
- Radius defaults retained: LMCA/LAD/LCX diameters 4.0/2.6/2.4 mm.
- Motion remains a conservative parametric LCA motion model; no patient-specific FSI or clinical validation is claimed.

## Acceptance interpretation

The production anatomy predicate is a transparent dataset-derived engineering gate. Stricter absolute inferior reach, sustained descent, and absolute lateral reach remain descriptive warnings because they were not the generator's hard acceptance rules. This distinction is now explicit in figures, CSVs and report prose.

## Literature used as sanity context

- PMID 37829965: LMCA/LAD/LCX morphometry and diameters.
- PMID 36944018: LMCA length and LAD–LCX angle cadaveric measurements.
- PMID 24098082: landmark-dependent coronary/cardiac displacement.
- PMID 8043342: early-diastolic maximum lumen area and reduced plaque-segment cyclic change.
- PMID 7611122: contrary observation of systolic coronary lumen expansion.

All external values are descriptive sanity checks; none is imposed as a clinical acceptance limit.
