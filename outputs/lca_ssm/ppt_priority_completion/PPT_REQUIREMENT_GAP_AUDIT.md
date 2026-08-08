# Original PPT requirement gap audit

This audit was completed before implementing the new PPT-priority measurement pipeline. The later heart-support-surface work is preserved as an extension and is not treated as a substitute.

| PPT requirement | Status before this increment | Existing module/output | Reuse decision / gap |
|---|---|---|---|
| Population NIfTI discovery | DONE | `nii files/` | Programmatic discovery retained |
| Centerline extraction | DONE | `lca_ssm_label_adapter.py; outputs/lca_ssm/raw_cases` | 191 extracted; 9 documented failures |
| Landmark extraction | DONE | `build_lca_population_ssm.py::select_landmarks` | Endpoints and bifurcation reused as exact source samples |
| Branch identification | DONE | `build_lca_population_ssm.py::classify_daughters` | RAS multi-signal inference; not ground truth |
| Coronary SVD plane | PARTIAL | `build_lca_population_ssm.py::process_case` | Reusable SVD; new run adds fixed RAS basis/sign convention |
| LAD SVD plane | PARTIAL | `lca_ssm_planes.py::fit_plane_svd` | Reusable SVD; new run adds population-comparable basis |
| Coronary ellipse | INCONSISTENT_WITH_FINAL_DEFINITION | `ellipse_references` | Prior constrained landmark arc was diagnostic; new all-support actual ellipse required |
| LAD ellipse | INCONSISTENT_WITH_FINAL_DEFINITION | `ellipse_references` | Prior constrained landmark arc was diagnostic; new all-source actual ellipse required |
| Ellipse a/b/tilt | PARTIAL | `population_statistics` | Existed for reference arcs; circular tilt statistics missing |
| Landmark ellipse angles | PARTIAL | `ellipse reference JSON` | Needed fixed global convention and direct endpoint exports |
| Branch angular extents | PARTIAL | `ellipse reference JSON` | Needed ordered full-centerline unwrapped extents |
| Ellipse residual/noise | PARTIAL | `ellipse reference JSON` | Needed signed in-plane/out-of-plane per-point population CSV |
| Population table | MISSING | `none in exact PPT schema` | One-row-per-patient table added by this increment |
| Statistical distributions | PARTIAL | `population_statistics` | Needed IQR and circular summaries |
| Statistical covariance/dependence | PARTIAL | `parameter correlation arrays` | Needed readable CSV and figure |
| Ellipse parameter sampling | MISSING | `future boundary` | Intentionally deferred by user |
| Landmark sampling | MISSING | `future boundary` | Intentionally deferred by user |
| Control-point distribution on ellipse arcs | MISSING | `future boundary` | Intentionally deferred by user |
| Residual/noise sampling | MISSING | `future boundary` | Intentionally deferred by user |
| Bifurcation connection | PARTIAL | `existing topology code` | Real topology exists; synthetic connection deferred |
| B-spline interpolation | DONE | `existing LCA/RCA topology modules` | Available for later reuse; not invoked |
| Validation | PARTIAL | `existing validation utilities` | Measurement validation implemented; synthetic validation deferred |

## Scope boundary

The current increment stops after population statistics. Sampling, synthetic landmarks/control points, synthetic noise, bifurcation generation, B-spline generation, VTK generation of synthetic trees, and real-versus-generated comparison are deliberately not implemented.
