# Coronary4D final release artifacts

The final technical package is complete. The presentation deck is intentionally
deferred for a later task, as recorded in `RELEASE_MANIFEST.json`; use
`FINAL_PROJECT_REPORT.docx` as the verified source for that future deck.

The primary ParaView deliverables are:

- `final_presentation/final_static_tree.vtm`: canonical static centerlines.
- `final_presentation/final_tree_with_scaffold.vtm`: canonical centerlines and
  ellipsoid scaffold.
- `final_presentation/cine.pvd`: canonical closed 4D animation.
- `final_presentation_52/`: the same presentation contract for all 52 accepted
  trees, plus separate healthy/diseased tapered meshes and mesh+centerline
  overlays.

The `demo_cases` directory contains four portable cases generated on
the same seeded anatomy:

- `healthy`: radius taper, cardiac motion, and pulsatility without a lesion.
- `focal_lad`: one smooth 65% focal LAD radius stenosis.
- `diffuse_lcx`: one smooth 45% diffuse LCX radius stenosis.
- `tandem_lad`: two smooth 55% focal LAD stenoses composed as tandem disease.

Large standalone Plotly HTML copies are intentionally omitted from the final
branch; run the `visualize --standalone-html` command when a self-contained
interactive file is needed.

Each case passed static anatomy generation, disease validation, native 4D
validation, fixed-size export validation, checksum generation, and VTK
multiblock readback. Open a case's `preview.png` for the presentation summary or
open `vtk/cine.pvd` in ParaView for the closed 10-frame cardiac cycle.

The final population run represents all 52 eligible real baselines once. All
52 generated trees passed, and all 50 real-versus-generated distribution checks
passed at the calibrated PCA innovation scale of 0.04. The demonstration motion
uses 14% radial contraction, 10% longitudinal shortening, and 10-degree torsion.

The `reports` directory contains one quantitative DOCX report per case. Each
report covers branch dimensions, distance-metric tortuosity, anatomical role
checks, stenosis severity and equivalent area reduction, 4D motion, local and
global pulsatility, acceptance evidence, limitations, and primary literature
context. Machine-readable audit results are stored as
`demo_cases/<case>/quantitative_validation.json`; the matching plot is
`visualizations/quantitative_motion_pulsatility.png`.

For the clearest presentation, start with `final_presentation/front.png`, then
open `final_presentation_52/tree_0015/diseased_mesh_with_centerlines.vtm` to
show taper and stenosis. The controlled disease comparison remains at
`demo_cases/disease_mode_comparison.png`,
then open `demo_cases/focal_lad/visualizations/validation_dashboard.png`. The
four cases include looping `cardiac_cycle.gif` files plus tortuosity analysis;
interactive HTML can be regenerated on demand with the command above.

Verify a copied deliverable from the repository root:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator verify `
  --input-dir submission_release\demo_cases\focal_lad
```

The statistical model is explicitly LCA-only: LMCA, LAD, and LCX. It does not
claim a population-derived RCA. These artifacts are research/engineering
prototypes and are not clinically validated.
