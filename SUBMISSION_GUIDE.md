# 4D coronary tree generator — submission guide

## Deliverable status

This repository provides a reproducible, validated **LCA-only** research
generator. It samples LMCA/LAD/LCX anatomy from the frozen 52-case eligible
statistical cohort, applies controlled focal/diffuse/tandem stenosis, adds
ellipsoid-relative cardiac motion and phase-dependent radius pulsatility, and
exports a portable 4D tensor plus ParaView files.

The project deliberately does not label the unresolved disconnected RCA data
as a population model. The separate RCA utilities are retained for future
work, but RCA is outside this release's statistical-generation contract.

## Five-minute run

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_integrated_4d_generator -v
.\.venv\Scripts\python.exe -m vessel_tree_generator demo --output-dir submission_release\demo_cases --clean
.\.venv\Scripts\python.exe -m vessel_tree_generator verify --input-dir submission_release\demo_cases\focal_lad
.\.venv\Scripts\python.exe -m vessel_tree_generator visualize `
  --input-dir submission_release\demo_cases\focal_lad --clean --standalone-html
.\.venv\Scripts\python.exe -m vessel_tree_generator compare `
  --input-dir submission_release\demo_cases
```

Generate one custom case:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator generate `
  --output-dir my_focal_case --preset focal --branch LAD `
  --position 0.45 --length 0.12 --severity 0.65 --seed 20260822 --clean
```

Or supply a disease file:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator generate `
  --output-dir my_case --config examples\disease_tandem_lad.json --clean
```

For a clean Python environment, create a virtual environment and run
`python -m pip install -r requirements.txt`. An editable installation
(`python -m pip install -e .`) also exposes the `coronary4d` command.

## Python API

```python
from vessel_tree_generator import CoronaryTreeGenerator, stenosis_config
from vessel_tree_generator.export import export_case

generator = CoronaryTreeGenerator()
disease = stenosis_config(
    branch_id="LAD",
    position=0.45,       # normalized branch arc length
    length=0.12,         # normalized branch arc length
    severity=0.65,       # fractional radius reduction
    lesion_type="focal",
)
case = generator.generate_case(disease)
export_case(case, "my_case")
```

`generator.generate_tree(phase=0.35, disease_config=disease)` returns one exact
requested phase. For controlled comparisons, call `sample_reference()` once
and pass that reference to multiple `generate_case()` calls.

## Output contract

Each exported case contains:

- `geometry_cine.npy`: `(phase, branch, point, 4)` tensor.
- `geometry_static.npy`: first/reference phase `(branch, point, 4)`.
- `coronary_tree_4d.npz`: compressed tensor, phase/time arrays, and labels.
- `metadata.json`: seed, model settings, disease, motion, pulsatility, and validation.
- `graph.json`: LMCA → LAD/LCX topology and branch attributes.
- `vtk/cine.pvd`: time-series entry point for ParaView.
- `preview.png`: fixed front views, radius profile, motion, and pulsatility QC.
- `manifest.json`: shapes, checksums, validation status, and VTK readback status.

Run the `verify` command after copying a case to independently check its
checksums, array shape, positive radii, exact topology, and VTK readability.

## Visual validation tools

Every case can produce a `visualizations/` directory containing:

- `validation_dashboard.png`: 3D, fixed-front and crown anatomy views; lesion
  overlay; healthy/diseased radii; cardiac motion; pulsatility; topology checks;
  and branch tortuosity summary.
- `tortuosity_analysis.png`: discrete curvature along LMCA, LAD and LCX plus
  arc/chord and total-turn metrics.
- `cardiac_cycle.gif`: fixed-camera looping 3D animation across all phases.
- `interactive_tree.html`: rotatable and zoomable 3D tree with cardiac phase
  slider, Play/Pause, point hover, radius-sized markers, and disease coloring.
- `visualization_metrics.json`: machine-readable values used by the plots.

Use `--standalone-html` when the interactive viewer must work without internet;
it embeds Plotly and is consequently about 5 MB. Without the flag, the much
smaller HTML loads Plotly from its CDN. ParaView remains the best tool for VTK
inspection: open `vtk/cine.pvd`, color by `radius_mm` or
`disease_reduction_fraction`, and play the time series.

The `compare` command produces `disease_mode_comparison.png`, placing healthy,
focal LAD, diffuse LCX, and tandem LAD cases side-by-side on identical anatomy.

The final axis order is `[x_mm, y_mm, z_mm, radius_mm]`; branch order is
`[LMCA, LAD, LCX]`. The cardiac period is derived from heart rate, and every
phase contains the same points in the same order.

## Disease and 4D behavior

- Disease changes radius only; it does not bend or replace centerlines.
- Focal lesions use a smooth compact profile; diffuse lesions use a smooth
  plateau; tandem disease composes multiple focal lesions.
- The configured stenosis severity is fractional **radius reduction**.
- The phase-zero radius is the diseased reference radius.
- Healthy epicardial lumen pulsatility defaults to 3% systolic expansion;
  maximal-lesion pulsatility defaults to 35% of the healthy amplitude, with a
  smooth transition. This is a scalar research approximation, not a wall model.
- Motion defaults are 14% radial contraction, 10% longitudinal shortening,
  10-degree base-to-apex torsion, and peak systole at normalized phase 0.35.
- Static generation uses a case-matched empirical baseline with a conservative
  PCA innovation scale of 0.04, calibrated on the full 52-case eligible cohort.
- All defaults are explicit research-prototype parameters, not learned clinical
  claims. Every parameter can be audited in `metadata.json`.

## Validation and limitations

Generation rejects invalid static anatomy before disease or motion. The 4D
validator checks mandatory branches, exact bifurcation continuity at every
phase, phase-zero identity, temporal point correspondence, finite positive
radii, configured pulsatility bounds, and disease validation. Export then
reads the first and last VTK multiblock files back from disk.

Known limitations:

- The population-derived model covers LCA only; no population-derived RCA.
- Radius taper, motion amplitudes, and compliance are mechanistic defaults,
  not patient-estimated parameters.
- No flow/CFD, vessel wall, side branches, or clinical outcome prediction.
- Engineering tests and anatomical gates do not constitute clinical validation.

## Population and final-audit commands

```powershell
.\.venv\Scripts\python.exe pipeline.py generate --count 52 --pca-scale 0.04 --clean
.\.venv\Scripts\python.exe pipeline.py validate --clean
.\.venv\Scripts\python.exe pipeline.py novelty-validate
.\.venv\Scripts\python.exe pipeline.py holdout-validate
.\.venv\Scripts\python.exe pipeline.py audit-final
```

The last three commands write non-identity/nearest-training metrics, a
source-grouped five-fold internal holdout (never called external validation),
the verified 200 → 191 → 181 → 65 → 52 funnel, original two-plane/two-ellipse
evidence, PCA recomputation, VTK readback, and release figures. Ten stored 4D
frames mean nine independent geometric positions plus the repeated phase-1
closure frame.

## Presentation path

For a demonstration, open `submission_release/demo_cases/focal_lad/vtk/cine.pvd`
in ParaView and press Play. Explain that all four provided cases use the same
seeded anatomy, so the visible difference is controlled disease rather than a
different sampled patient. Use `preview.png` to explain the fixed cardiac front
view, radius reduction, cyclic motion, and cyclic radius behavior.
