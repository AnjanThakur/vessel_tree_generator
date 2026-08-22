# Final implementation truth audit

This audit classifies code by direct inspection and machine-readable evidence. PASS means engineering evidence, not clinical validation.

| Component | Classification | Source | Function/class | Input | Output | Tests | Evidence | Report accuracy | Action |
|---|---|---|---|---|---|---|---|---|---|
| Source extraction | VERIFIED_IMPLEMENTED | ppt_priority_population_model.py | extract/load protected centerlines | NIfTI/archives | source centerlines + landmarks | tests + protected integrity | ppt_priority_completion | Accurate | None |
| Two measured planes | VERIFIED_IMPLEMENTED | lca_ssm_planes.py | fit_plane_svd | source XYZ | centroid/normal/residual | plane tests | 191 parameter rows | Accurate | None |
| Two actual ellipse fits | VERIFIED_IMPLEMENTED | ppt_priority_population_model.py | fit_actual_ellipse | 2-D projected points | a,b,tilt,residuals | PPT compliance tests | 191 fits / 77,337 residuals | Accurate | None |
| Measured planes vs cardiac frame | IMPLEMENTED_BUT_UNDERDOCUMENTED | lca_ssm_planes.py / run_person1_week1_pipeline.py | plane SVD / cardiac frame | measured normals | measured planes + orthonormal derived frame | frame validation | 191 frame PASS rows | Now clarified | Documentation |
| Support ellipsoid | VERIFIED_IMPLEMENTED | ellipsoid_model.py | build_support_ellipsoid | measured scaffold | positive a,b,c | axis validation | population_ellipsoid_parameters.csv | Accurate | Robust statistics added |
| Surface representation | IMPLEMENTED_BUT_UNDERDOCUMENTED | surface_projection.py | project_point_to_surface | cardiac XYZ | angular/radial u,v,offset + local basis | round-trip test | surface coordinates | Nearest-point wording removed | Rename/document claim |
| Fixed correspondence | VERIFIED_IMPLEMENTED | fixed_representation.py | arc_length_resample | full-resolution analysis copies | 5/12/10 points | endpoint/topology tests | 52x27x3 NPZ | Accurate | 10-case visual QC |
| Joint PCA | VERIFIED_IMPLEMENTED | run_person1_week1_pipeline.py | fit PCA by SVD | 52x81 local vectors | 13-mode model | independent recomputation | frozen PCA NPZ | Accurate | Independent audit |
| Empirical path smoothing | INCONSISTENT | generation/surface_path_generator.py | interpolate_bspline_points | fixed samples | dense B-spline | 19 generation tests + 52-case trial | 52/52 deterministic trial | Previously called B-spline but used PCHIP | Converted exact stable Hermite form to explicit BSpline basis |
| Static generator | VERIFIED_IMPLEMENTED | tree_assembler.py | assemble | joint empirical scaffold + PCA innovation | LMCA/LAD/LCX | full validator | 52 accepted trees | Accurate as bootstrap generator | Novelty quantified |
| Pure PPT ellipse-arc generator | DOCUMENTED_ONLY | historical measurement outputs | N/A | ellipse statistics | N/A | N/A | measurement evidence only | Advanced model generalizes rather than literally duplicates | Not added; advanced mode retained and limitation explicit |
| RCA population model | MISSING | N/A | N/A | unresolved candidate RCA | N/A | N/A | candidate-only evidence | Correctly excluded | Await validated labels |
| Side branches | MISSING | N/A | N/A | no reliable labels | N/A | N/A | none | Correctly excluded | Future work |
| Radius/taper | VERIFIED_IMPLEMENTED | vessel_tree_generator/radius.py | assign_radius_profiles | static LCA | positive tapered radii | staged/integrated tests | demo arrays | Accurate as parametric | None |
| Disease | VERIFIED_IMPLEMENTED | disease.py | apply_disease | healthy radii | focal/diffuse/tandem radii | identity tests | four demos | Accurate as radius-only prototype | Audit added |
| 4D motion/pulsatility | VERIFIED_IMPLEMENTED | motion.py / pulsatility.py | generate_cine | static geometry/radii | closed 4D cycle | staged/integrated tests | 4 demos + 520 snapshots | Accurate as parametric synthetic motion | Phase convention clarified |
| Internal holdout | PARTIAL | final_validation_audit.py | holdout_audit | 52 eligible sources | 5-fold representation evidence | leakage assertions | holdout CSV/JSON | New; not external validation | Retain limitation |

## Verified numerical anchors

- Original measurement cases: 191; residual records: 77,337.
- PCA: 52 x 81; 13 modes; 95.5469% variance.
- Projection round-trip: max 2.170e-12 mm across 156,690 records.
- Novelty: 0 exact duplicates; mean baseline displacement 0.9337 mm.
- Holdout: 5 folds; source leakage 0; representation-level generated anatomy acceptance 78.8%.
- Eligible ellipsoids: 52; all positive/finite = True.
