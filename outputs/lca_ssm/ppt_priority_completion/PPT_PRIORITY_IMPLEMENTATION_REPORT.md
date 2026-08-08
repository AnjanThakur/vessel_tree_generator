# PPT-priority implementation report

## Result

- NIfTI volumes discovered: 200
- Extracted centerline cases available: 191
- Documented extraction failures: 9
- Cases with both SVD planes: 191
- Cases with both actual ellipses: 191
- Cases entering statistics: 191
- Cases using inferred RCA candidate + LCX crown support: 189
- Cases using flagged LCX-only crown approximation: 2
- Pointwise residual records: 77,337
- New presentation: `presentation/LCA_PPT_Priority_Two_Plane_Two_Ellipse_Generative_Model.pptx` (18 slides; zero layout QA errors/warnings)

The original PPT plane and ellipse steps are now separate in both code and outputs. The two bounded ovals drawn for visualization are sampled from fitted ellipse parameters, not visual representations of infinite SVD planes.

## Fit-quality interpretation

Full ellipses are being inferred from arterial support that may cover only part of an ellipse. This produces long-tailed axis estimates and must remain visible rather than being silently trimmed. Forty-five crown fits have absolute Euclidean RMSE greater than 25% of their fitted semi-major axis; three LAD fits have axis ratio above 10. They remain in the population table because the source is valid and the task forbids discarding difficult anatomy merely to improve RMSE. Median, IQR, P5/P95, maximum residual and per-case warnings are therefore essential alongside mean and normalized summaries.

## Preserved extension work

`outputs/lca_ssm/stage1_heart_scaffold_vtk/` and `outputs/lca_ssm/stage1_parametric_heart_surface/` were hash-checked before and after this run and were not changed. They remain extended anatomical support-surface work, separate from the original PPT population model.

## Deliberate stopping point

No parameter sampler, landmark sampler, synthetic residual/noise model, generated control points, generated bifurcation, B-spline synthetic tree, synthetic VTK, or real-versus-generated comparison was implemented in this increment.
