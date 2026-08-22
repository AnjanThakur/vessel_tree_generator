# Original PPT compliance matrix

| Requirement | Implementation/source | Output evidence | Current status | Final action |
|---|---|---|---|---|
| ~200 NIfTI population | ppt_priority_population_model.py | population_case_status.csv | VERIFIED | 200 discovered |
| Centerlines + landmarks | source-preserving extraction | 191 successful rows | VERIFIED | No source changes |
| Two measured planes | fit_plane_svd | coronary/LAD normals and residuals | VERIFIED | Preserve nonorthogonal measurements |
| Two fitted ellipses | fit_actual_ellipse | crown/LAD a,b,tilt | VERIFIED | Actual planar ellipse fit, not SVD patch |
| Landmark theta / branch extents | ellipse-frame angular calculations | population parameter CSV/JSON | VERIFIED | Circular statistics retained |
| Per-point deviations | ellipse reference residuals | 77,337 rows | VERIFIED | Descriptive noise evidence |
| Population statistics | robust + circular summaries | population_statistics.json | VERIFIED | Long tails not assumed Gaussian |
| Sample ellipse-arc generator | advanced ellipsoid/PCA generalization | surface/PCA generator | EXTENDED_NOT_LITERAL | Report distinction explicitly |
| Connect at bifurcation | TreeAssembler exact snap | topology error 0 | VERIFIED | Exact array equality |
| B-spline interpolate | interpolate_bspline_points | 52-case equivalence trial | VERIFIED_AFTER_FIX | Explicit shape-preserving cubic BSpline basis |
| Validate | TreeValidator + integrated validation | cohort/demo/VTK audits | VERIFIED | Engineering/anatomical plausibility only |
