# Final codebase cleanup audit

## Classification

| Class | Files / areas | Decision |
|---|---|---|
| FINAL_PRODUCTION_REQUIRED | `vessel_tree_generator/`, production generation/surface/PCA/motion modules, frozen statistics, accepted cohort | Retained |
| SHARED_UTILITY_REQUIRED | ellipse measurement, cardiac-frame, surface projection, VTK export, validation utilities | Retained because production/evidence imports use them |
| DOCUMENTATION_ONLY / MEASUREMENT EVIDENCE | tracked 191-case two-plane/two-ellipse tables, residuals, figures and integrity records | Retained as source/statistical evidence, not a competing generator |
| EXPERIMENT_ONLY | `ppt_exact_generator.py`, refinement module, two runners and two test files | Removed from production/import/test paths and placed in an ignored local research archive |
| GENERATED_EXPERIMENT_ARTIFACT | `submission_release/ppt_exact_generator*` | Removed from canonical release and placed in the same ignored local research archive |

## Production result

- Cohort: 52/52 readable, topology-valid and production-anatomy-valid.
- Presentation medoid: `tree_0015`; warning count: 0.
- Protected raw archives: 191/191 hash matches.
- Source-coordinate change: 0.0 mm.
- Source segment-length change: 0.0 mm.
- Regression families: 73 main production, 21 focused generation, 9 design-alignment; 103 total, 0 failed.
- Final code path: `vessel_tree_generator.CoronaryTreeGenerator` / `python -m vessel_tree_generator`.
- `FINAL_PROJECT_REPORT.docx` and presentation slides were intentionally left unchanged at the user's request.
