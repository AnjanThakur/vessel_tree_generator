# Disease-aware 4D Coronary LCA Tree Generator

This repository is a usable research pipeline for generating a labeled
`LMCA + LAD + LCX` tree with `(x, y, z, radius)` at each point and across cardiac
phase. It combines the audited LCA statistical shape model with controlled
focal/diffuse/tandem stenosis, cardiac motion, reduced compliance at lesions,
reproducible seeds, validation, NumPy/JSON output, and animated ParaView export.

The primary submission interface is:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator demo `
  --output-dir submission_release\demo_cases --clean
```

For one custom focal case:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator generate `
  --output-dir my_case --preset focal --branch LAD `
  --position 0.45 --length 0.12 --severity 0.65 --clean
```

Create a complete visual validation pack for that case:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator visualize `
  --input-dir my_case --clean --standalone-html
```

This writes a quantitative dashboard, tortuosity/curvature plot, looping 4D
cardiac-cycle GIF, metrics JSON, and rotatable interactive HTML with phase
slider, Play/Pause controls, radius-sized markers, and disease-colored points.

Run the independent anatomy, disease, motion and local/global pulsatility audit:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator audit `
  --input-dir submission_release\demo_cases --release
```

The curated release also includes one rendered-and-checked quantitative DOCX
report per tree in `submission_release/reports/`.

See [SUBMISSION_GUIDE.md](SUBMISSION_GUIDE.md) for the API, disease convention,
output schema, ParaView workflow, validation contract, and honest limitations.

The underlying statistical workflow starts from protected centerlines extracted
from the approximately 200 supplied NIfTI label volumes, resolves LMCA daughter
identities in a common RAS/cardiac frame, and fits the LCA-only shape model.

The primary generated topology is **LMCA + LAD + LCX**. RCA is deliberately not
included because the available disconnected RCA candidates are not resolved
ground truth.

## Current verified population

- 200 source label volumes are available locally.
- 191 cases have protected raw LCA centerline records.
- 181 daughter assignments pass the multi-signal confidence gate.
- 65 resolved cases pass both source and scaffold core-anatomy gates; 13 then
  fail the predeclared ellipsoid-quality contract, leaving 52 PCA cases.
- The primary representation has 27 points: LMCA 5, LAD 12, and LCX 10.
- The PCA matrix is 52 x 81; 13 modes retain 95.55% cumulative variance.
- The final reference run generated 52/52 accepted static trees in 64 attempts,
  representing every eligible source baseline exactly once.
- All 50 real-versus-generated population comparisons pass with no descriptive
  warnings at the validated PCA innovation scale of 0.04.
- Optional motion generated 52 trees x 10 stored frames: nine independent
  geometric positions plus the repeated phase-1 closure frame, with exact
  phase-0/phase-1 identity and continuous LMCA-to-daughter junctions.

These are engineering validation results, not claims of clinical validity.
Motion amplitudes and radius tapers are explicit prototype defaults, not learned
population parameters.

## Statistical-model maintenance commands

Run commands from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe pipeline.py compute-stats --clean
.\.venv\Scripts\python.exe pipeline.py generate --clean
.\.venv\Scripts\python.exe pipeline.py validate --clean
.\.venv\Scripts\python.exe pipeline.py motion
.\.venv\Scripts\python.exe pipeline.py export
.\.venv\Scripts\python.exe pipeline.py audit-final
.\.venv\Scripts\python.exe pipeline.py novelty-validate
.\.venv\Scripts\python.exe pipeline.py holdout-validate
```

Or rebuild the complete statistical pipeline:

```powershell
.\.venv\Scripts\python.exe pipeline.py run-all --clean
```

Use `.\.venv\Scripts\python.exe pipeline.py --help` for all configurable
paths and sampling parameters.

The independent final audit writes the verified cohort funnel, original-PPT
compliance package, PCA recomputation, novelty analysis, three-strategy
comparison, source-grouped five-fold internal holdout, VTK readback, and final
validation figures under `submission_release/final_audit/` and
`submission_release/final_validation/`.

## Canonical outputs

```text
outputs/lca_ssm/
|-- raw_cases/                  # immutable extracted source records (local)
|-- lca_population_model/       # audit, assignments, statistics, frozen PCA
|-- lca_population_cohort/      # accepted static trees and previews
|-- lca_population_motion/      # optional 4D prototype motion
`-- lca_population_export/      # fixed-size static/cine XYZ-radius arrays
```

New integrated 4D case exports additionally contain `geometry_cine.npy`,
`metadata.json`, `graph.json`, `manifest.json`, a QC preview, and `vtk/cine.pvd`.
The demo command creates healthy, focal, diffuse, and tandem cases on exactly
the same seeded anatomy for a controlled comparison.

The repository tracks the compact frozen generator package, run summaries, QC
images, and one complete representative tree. Bulk source data, per-case trial
runs, and full generated cohorts remain ignored.

## Validation policy

Training eligibility requires a confident multi-signal daughter assignment;
root radius or a single direction vector is never sufficient. Hard generation
checks enforce exact LMCA/LAD/LCX topology, an LMCA shorter than both daughters,
LAD-dominant inferior/apical course, an LCX crown-like lateral course, 3D
continuity, nonlocal self-clearance, and inter-branch clearance. Central
population intervals are warnings; observed hard bounds and core anatomy
violations reject a candidate.

Full-centerline fit errors remain descriptive because the generated vessel is a
smooth anatomical scaffold, not a point-by-point reproduction of segmentation
noise.

## Repository layout

- `pca_ssm_vessel_tree_generator/`: active extraction, alignment,
  surface-relative model, PCA, generation, motion, and export code.
- `vessel_tree_generator/`: public integrated 4D API, disease/pulsatility,
  validation/audit, CLI, visualization, and portable export.
- `examples/`: ready-to-run disease configurations.
- `tests/`: deterministic staged tests and pipeline integration tests.
- `lca_vessel_tree_generator/`: earlier LCA topology utilities retained for
  reference and compatibility.
- `rca_vessel_tree_generator/`: separate RCA/primitive-vessel utilities; not
  part of the canonical LCA statistical model.

See `pca_ssm_vessel_tree_generator/generation/README.md` for the generation
contract and artifact details.
