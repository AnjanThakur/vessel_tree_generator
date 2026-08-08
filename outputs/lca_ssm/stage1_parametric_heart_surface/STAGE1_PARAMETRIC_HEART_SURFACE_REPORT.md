# Stage-1 parametric anatomical support surface report

## 1. Objective
Create a generator-ready, dataset-derived anatomical support scaffold without modifying the source coronary centerlines.

> This is a dataset-derived parametric anatomical support scaffold, not a patient-specific myocardial reconstruction.

## 2. Input source files
- LMCA: `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_heart_scaffold_vtk\189.label\01_source_geometry\LMCA_centerline.vtp` (42 points)
- LAD: `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_heart_scaffold_vtk\189.label\01_source_geometry\LAD_centerline.vtp` (352 points)
- LCX: `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_heart_scaffold_vtk\189.label\01_source_geometry\LCX_centerline.vtp` (187 points)
- RCA: `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_heart_scaffold_vtk\189.label\01_source_geometry\RCA_candidate_centerline.vtp` (249 points)
- Saved measurements: `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_heart_scaffold_vtk\189.label\measurements.json`
- Saved validation: `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_heart_scaffold_vtk\189.label\validation.json`

## 3. Primary case
`189.label` was used. No alternate case was substituted.

## 4. Source point counts
- LMCA: 42
- LAD: 352
- LCX: 187
- RCA: 249

## 5. Plane fitting equations
For source points `p_i`: `C = mean(p_i)`, `Q_i = p_i - C`, `U,S,Vᵀ = SVD(Q)`, `n = Vᵀ[-1]`, and `n · (x - C) = 0`.
All relevant unchanged source-support points are used. Measurement planes pass through their true centroids.
The infinite fitted planes are displayed as support-derived bounded elliptical disks, following the `coronARY_SSM.pdf` plane convention; the oval boundary is not a second fit.
- Coronary plane role: LCX plus inferred RCA crown ring along the AV groove.
- Interventricular plane role: LAD descent toward the apex.

## 6. Plane statistics
- Coronary centroid-SVD plane RMSE: 1.630860 mm
- LAD centroid-SVD plane RMSE: 3.427237 mm
- Acute plane-normal separation: 65.563755°

## 7. Crown scaffold measurements
- Scaffold crown span: 98.515036 mm
- Scaffold crown depth: 71.787929 mm
- Coronary/crown reference derives from saved LCX plus inferred-RCA crown-support geometry.

## 8. Long-axis scaffold measurements
- Scaffold long-axis span: 93.192528 mm
- The long-axis/apical reference is supported by the unchanged LAD trajectory.

## 9. Parametric support-surface parameters
- Ellipsoid exponent: n = 2.0
- Semi-axes `(a crown, b depth, c long-axis)`: (49.257518, 35.893964, 46.596264) mm
- Canonical origin: [-16.072938206730925, 27.968819544333968, -51.05350074279319] mm
- Parameters are read from saved measurements; no 189.label dimensions are hard-coded.

## 10. Source-to-surface diagnostics
- LMCA: mean 13.445 mm; median 13.747 mm; RMSE 13.547 mm; P95 16.090 mm; max 16.392 mm
- LAD: mean 22.824 mm; median 23.836 mm; RMSE 23.054 mm; P95 27.146 mm; max 27.563 mm
- LCX: mean 8.001 mm; median 7.663 mm; RMSE 8.548 mm; P95 14.314 mm; max 16.392 mm
- RCA: mean 8.969 mm; median 8.796 mm; RMSE 10.515 mm; P95 20.000 mm; max 21.343 mm
These distances are descriptive only and are never used to move, project, or reject source centerline points.

## 11. Source-integrity proof
- Maximum source-coordinate change: 0.000e+00 mm
- Maximum original segment-length change: 0.000e+00 mm
- Maximum rigid-transform segment error: 1.066e-14 mm
- Maximum VTK canonical-coordinate difference: 0.000e+00 mm
- Source file hashes unchanged: True
- Overall validation: **PASS**

## 12. VTK files generated
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\01_source_geometry\LMCA_centerline.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\01_source_geometry\LAD_centerline.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\01_source_geometry\LCX_centerline.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\01_source_geometry\RCA_candidate_centerline.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\01_source_geometry\bifurcation.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\02_measurement_planes\coronary_centroid_SVD_plane.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\02_measurement_planes\LAD_centroid_SVD_plane.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\03_reference_ellipses\coronary_crown_reference_ellipse.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\03_reference_ellipses\long_axis_apical_reference_ellipse.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\03_reference_ellipses\crown_axis.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\03_reference_ellipses\secondary_crown_axis.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\03_reference_ellipses\apex_axis.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\04_parametric_surface\parametric_heart_support_surface.vtp`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\03_reference_ellipses\heart_scaffold.vtm`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\01_source_geometry\source_centerlines.vtm`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\05_overlay\source_plus_scaffold.vtm`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\05_overlay\source_plus_parametric_surface.vtm`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\05_overlay\stage1_complete_anatomical_model.vtm`

## 13. PPT figures generated
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\01_original_dataset_centerlines.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\01_original_dataset_centerlines.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\02_centroid_svd_measurement_planes.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\02_centroid_svd_measurement_planes.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\03_two_ellipse_anatomical_scaffold.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\03_two_ellipse_anatomical_scaffold.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\04_3d_parametric_anatomical_support_surface.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\04_3d_parametric_anatomical_support_surface.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\05_unchanged_centerlines_on_support_surface.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\05_unchanged_centerlines_on_support_surface.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\06_superior_crown_view.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\06_superior_crown_view.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\07_front_long_axis_view.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\07_front_long_axis_view.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\08_lateral_view.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\08_lateral_view.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\09_observed_vs_extrapolated_reference_geometry.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\09_observed_vs_extrapolated_reference_geometry.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\10_source_to_surface_distance_diagnostics.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\10_source_to_surface_distance_diagnostics.svg`
- `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\11_stage1_technical_validation_summary.png` and `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\189.label\06_diagnostics\11_stage1_technical_validation_summary.svg`
- Presentation: `C:\Internship\vessel_tree_generator\outputs\lca_ssm\stage1_parametric_heart_surface\presentation\LCA_Stage1_Dataset_Derived_Anatomical_Scaffold.pptx` (generated after geometric validation)

## 14. Limitations
- The inferred RCA candidate is not annotated RCA ground truth.
- Coronary-derived scaffold dimensions are not myocardial dimensions.
- Dashed reference-ellipse portions are extrapolated.
- The surface is a parametric anatomical support model, not a clinical reconstruction.
- No generative coronary geometry is present in Stage-1.

## 15. Next-stage architecture
Stages A/B are complete. Stage C should generate coronary trajectories directly in surface-relative `(theta, phi, offset)` coordinates; Stages D–H remain future work.

## 16. Exact regeneration commands
```powershell
python pca_ssm_vessel_tree_generator/build_parametric_heart_support_surface.py --case 189.label --reset-output
```
