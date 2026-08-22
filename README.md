# LCA Statistical Vessel Tree Generator

This repository contains a data-audited left-coronary-artery (LCA) statistical
shape pipeline. The canonical workflow starts from the protected centerlines
extracted from the approximately 200 supplied NIfTI label volumes, resolves the
LMCA daughter identities in a common RAS/cardiac frame, fits an LCA-only shape
model, generates validated static trees, and can optionally add prototype
cardiac motion and export fixed-size XYZ-radius arrays.

The primary generated topology is **LMCA + LAD + LCX**. RCA is deliberately not
included because the available disconnected RCA candidates are not resolved
ground truth.

## Current verified population

- 200 source label volumes are available locally.
- 191 cases have protected raw LCA centerline records.
- 181 daughter assignments pass the multi-signal confidence gate.
- 52 cases pass assignment, frame, ellipsoid, source-integrity, representation,
  and core anatomical checks and enter the PCA.
- The primary representation has 27 points: LMCA 5, LAD 12, and LCX 10.
- The PCA matrix is 52 x 81; 13 modes retain 95.55% cumulative variance.
- The reference run generated 25/25 accepted static trees in 30 attempts.
- Optional motion generated 25 trees x 10 phases with exact phase-0 identity
  and continuous LMCA-to-daughter junctions.

These are engineering validation results, not claims of clinical validity.
Motion amplitudes and radius tapers are explicit prototype defaults, not learned
population parameters.

## Canonical commands

Run commands from the repository root with the project virtual environment:

```powershell
.\.venv\Scripts\python.exe pipeline.py compute-stats --clean
.\.venv\Scripts\python.exe pipeline.py generate --clean
.\.venv\Scripts\python.exe pipeline.py validate --clean
.\.venv\Scripts\python.exe pipeline.py motion
.\.venv\Scripts\python.exe pipeline.py export
```

Or run the complete pipeline:

```powershell
.\.venv\Scripts\python.exe pipeline.py run-all --clean
```

Use `.\.venv\Scripts\python.exe pipeline.py --help` for all configurable
paths and sampling parameters.

## Canonical outputs

```text
outputs/lca_ssm/
|-- raw_cases/                  # immutable extracted source records (local)
|-- lca_population_model/       # audit, assignments, statistics, frozen PCA
|-- lca_population_cohort/      # accepted static trees and previews
|-- lca_population_motion/      # optional 4D prototype motion
`-- lca_population_export/      # fixed-size static/cine XYZ-radius arrays
```

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
- `tests/`: deterministic staged tests and pipeline integration tests.
- `lca_vessel_tree_generator/`: earlier LCA topology utilities retained for
  reference and compatibility.
- `rca_vessel_tree_generator/`: separate RCA/primitive-vessel utilities; not
  part of the canonical LCA statistical model.

See `pca_ssm_vessel_tree_generator/generation/README.md` for the generation
contract and artifact details.
