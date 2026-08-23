# Final codebase cleanup audit

## Final-deliverable classification

| Class | Files / areas | Decision |
|---|---|---|
| FINAL_PRODUCTION_REQUIRED | `vessel_tree_generator/`, production generation/surface/PCA/motion modules, frozen statistics, accepted cohort | Retained |
| SHARED_UTILITY_REQUIRED | cardiac-frame, surface projection, B-spline, VTK export and validation utilities | Retained because production imports use them |
| FROZEN_RUNTIME_DATA | compact 52-case generator statistics | Retained and SHA-256 verified |
| PRESENTATION_REQUIRED | canonical presentation, all-52 mesh/4D package, presentation center | Retained |
| HISTORICAL_BULK | intermediate Stage-1, person-1/person-2 and PPT-priority output trees | Excluded from final branch; retained on `latestt_branchh` at `7b21ce4` |
| EXCLUDED_SCOPE | disconnected RCA and obsolete LCA prototype shells | Excluded after required radius/disease/mesh modules were consolidated into the public package |

## Production result

- Cohort: 52/52 readable, topology-valid and production-anatomy-valid.
- Presentation medoid: `tree_0015`; warning count: 0.
- Protected raw archives: 191/191 hash matches.
- Source-coordinate change: 0.0 mm.
- Source segment-length change: 0.0 mm.
- Regression families: 72 main production, 21 focused generation, 9 design-alignment; 102 total, 0 failed.
- Final code path: `vessel_tree_generator.CoronaryTreeGenerator` / `python -m vessel_tree_generator`.
- Final presentation payload: 52/52 trees, seven entry files per tree and 520 closed-cycle frames.
- Historical paths and recovery instructions are listed in `docs/ARCHIVE.md`.
