# Anatomical LCA Reorientation MVP

## Why This Was Needed

The dataset-driven LCA control points were mathematically usable, but their 3D
layout did not always resemble heart-based coronary anatomy.

The MVP anatomical reorientation creates a rule-based LCA template where:

- LMCA remains a short trunk before bifurcation
- LAD descends downward toward the apex
- LCX curves laterally around the left side like a circumflex/crown vessel
- LAD and LCX are constrained to simple heart-landmark guide curves

This fixes only the 3D control-point layout. It does not change disease, radius,
tube surface, tight mesh, or bifurcation hub code.

## Literature-Based Justification

Zamir's coronary anatomy description supports the main rules used in this MVP.
The coronary arteries follow a functional heart-based layout rather than
arbitrary 3D paths. LAD descends toward the apex along the interventricular
groove, while LCX follows a circumflex route around the left side of the heart
in the atrioventricular/coronary plane. RCA follows a similar crown-like route
on the right side.

This supports the core template assumptions:

- LAD should have a strong downward/apex-directed component.
- LCX should curve laterally/circumferentially like a crown branch.
- LCX should not behave like a second LAD.
- LMCA should remain a short parent trunk before bifurcation.
- Main arteries such as LAD, LCX, and RCA behave as distributing vessels.

The literature also emphasizes that coronary anatomy varies significantly
between hearts. A true patient-specific clinical reconstruction would therefore
require reliable segmented coronary centerlines, heart-surface meshes, or
anatomical landmarks such as apex, base, and groove paths. Since the current
repo does not contain those clinical references, this module uses a rule-based
anatomical template as an MVP correction step.

Normal coronary structure is mainly an open tree, not a random set of
interconnected loops. Collateral connections are variable and often
disease-related, so this MVP preserves the existing topology:

```text
LMCA -> LAD + LCX
```

This module should therefore be described as an anatomically guided MVP
approximation, not a clinical reconstruction.

## Input Format

Input is the existing dataset control tree:

```text
LCA_tree_ctrl_points.npy
shape: N x 27 x 3
```

Each tree keeps the existing indexing:

```text
LMCA: indices 0-4
LAD:  indices 5-16
LCX:  indices 17-26
shared bifurcation: index 4
```

## Anatomical Rules

### Heart Landmark Frame

The anatomical template now uses a simple MVP heart-coordinate frame:

```text
ostium = LMCA start
LMCA bifurcation = shared parent/child branch point
Z axis = base-to-apex direction
X/Y plane = coronary/crown plane
apex = lower LAD-directed landmark
coronary/AV groove = LCX crown-plane guide
```

This is implemented in:

```text
LCA_topology_generator/heart_landmark_model.py
```

The goal is to make the template heart-guided rather than only viewer-direction
guided. It is still a rule-based MVP approximation, not a fitted heart-surface
or CTCA reconstruction.

### LMCA Short Trunk Rule

LMCA is generated as an ostium-to-bifurcation short trunk with slight curvature.

### LAD Apex-Groove Rule

LAD starts at the bifurcation and follows an apex-directed anterior
interventricular groove guide curve. It descends mainly along negative `Z`,
which is treated as the base-to-apex direction in this template.

### LCX Coronary-Groove Rule

LCX starts at the bifurcation and moves laterally around the left side in a
circumflex-style crown arc. It is constrained near the coronary/AV groove plane
so it does not behave like a second LAD.

## Scale Preservation

For each patient tree, the original LMCA, LAD, and LCX control-path lengths are
measured. The anatomical template branches are rescaled to preserve those branch
lengths as closely as possible.

## Outputs

The module writes:

```text
outputs/dataset_lca_anatomical/
  LCA_tree_ctrl_points_anatomical.npy
  anatomical_validation.json
  anatomical_summary.json
  trees/patient_XXXX/control_points_27x3_anatomical.npy
  trees/patient_XXXX/anatomical_control_points_3d.png
  trees/patient_XXXX/before_vs_after_control_points.png
  trees/patient_XXXX/anatomical_centerlines_preview.png
  trees/patient_XXXX/anatomical_control_points.vtk
  trees/patient_XXXX/anatomical_centerlines.vtk
```

By default, only source trees that pass the original dataset validation are
reoriented. Source-invalid trees are recorded in `anatomical_summary.json` under
`skipped_invalid_source_trees`.

Use `--include-invalid` only when you intentionally want to generate anatomical
template outputs for invalid source cases. Those outputs should be treated as
template-repaired geometry, not direct valid patient geometry.

The output control point format remains:

```text
27 x 3
```

The `.vtk` files are legacy VTK PolyData files for interactive inspection in
tools such as ParaView or 3D Slicer. They include point data:

```text
branch_id: LMCA=1, LAD=2, LCX=3
point_index: point index within each branch
```

Use `anatomical_control_points.vtk` to inspect the 27 control-point polylines.
Use `anatomical_centerlines.vtk` to inspect the smooth interpolated centerlines.

## Validation Checks

Validation checks include:

- shape is `27 x 3`
- no NaN or Inf values
- no zero-length branch
- no unexpected duplicate control points
- LAD starts at LMCA bifurcation
- LCX starts at LMCA bifurcation
- LAD has dominant downward negative-Z direction
- LAD follows the apex/interventricular guide curve
- LAD aligns with the base-to-apex axis
- LAD distal point is lower than LAD start
- LAD does not loop back upward
- LCX has dominant lateral/crown-like direction
- LCX follows the coronary/AV guide curve
- LCX stays near the coronary/crown plane
- LCX has limited vertical descent compared with LAD
- LMCA remains shorter than LAD and LCX
- topology remains LMCA -> LAD + LCX
- LAD/LCX separation angle is acceptable
- LAD/LCX non-bifurcation points do not overlap

Each per-patient validation JSON also stores quantitative metrics:

```json
{
  "lad_downward_score": 0.95,
  "lcx_lateral_score": 1.0,
  "lcx_vertical_score": 0.0,
  "heart_guided": {
    "lad_guide_distance_mean_ratio": 0.01,
    "lad_apex_axis_alignment": 0.95,
    "lcx_guide_distance_mean_ratio": 0.01,
    "lcx_coronary_plane_alignment": 0.99
  },
  "min_lad_lcx_separation_mm": 4.0,
  "anatomical_score": 0.9
}
```

These metrics are meant to reduce false-positive visual approval. LAD should be
vertical/apex-dominant, while LCX should be lateral/circumferential-dominant and
should not behave like a second LAD.

## How To Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m lca_vessel_tree_generator.LCA_topology_generator.anatomical_lca_reorientation
```

Or with `PYTHONPATH` style used by other LCA scripts:

```powershell
$env:PYTHONPATH='lca_vessel_tree_generator'
.\.venv\Scripts\python.exe -m LCA_topology_generator.anatomical_lca_reorientation
```

To force generation for source-invalid cases:

```powershell
.\.venv\Scripts\python.exe -m LCA_topology_generator.anatomical_lca_reorientation --include-invalid
```

## Current Limitations

- This is a rule-based anatomical approximation.
- It is not patient-specific clinical reconstruction.
- It should not be used to convert source-invalid patients into clinically valid
  patients without separate review.
- It does not infer heart pose from images.
- It does not fit vessels to a patient-specific heart surface.
- It does not use patient-specific apex, base, or groove landmarks.
- It does not model side branches.
- It preserves branch scale but not the original patient-specific 3D orientation.

The corrected anatomical control points are intended to remain compatible with
the existing centerline, radius, disease, tube, and tight mesh pipeline.

## Full Anatomical Pipeline Pass

The corrected anatomical control points can now be used as the base input for
the existing downstream modules:

```text
anatomical control points
-> spline centerlines
-> radius model
-> disease module
-> tube surface
-> tight mesh
-> validation reports
```

Run:

```powershell
$env:PYTHONPATH='lca_vessel_tree_generator'
.\.venv\Scripts\python.exe -m LCA_topology_generator.anatomical_lca_pipeline --input-dir outputs\dataset_lca_anatomical --output-dir outputs\dataset_lca_anatomical_pipeline --metadata-dir lca_vessel_tree_generator\LCA_branch_control_points\generated
```

The pipeline writes normal anatomical tree outputs under:

```text
outputs/dataset_lca_anatomical_pipeline/trees/patient_XXXX/
```

and disease-case outputs under:

```text
outputs/dataset_lca_anatomical_pipeline/disease/trees/patient_XXXX/
```

The generated summary is:

```text
outputs/dataset_lca_anatomical_pipeline/ANATOMICAL_PIPELINE_SUMMARY.md
```

The disease, tube surface, tight mesh, and hub connector modules are reused
unchanged.

## Controlled Synthetic Anatomical Data

A controlled synthetic augmentation pass is also available. It uses only valid
anatomical patient trees as base templates and applies small branch-level
perturbations:

- branch length scale: +/-5 percent by default
- LAD descent angle variation: +/-8 degrees by default
- LCX crown arc variation: +/-8 degrees by default
- small smooth control-point noise
- retry/reject if anatomical or existing centerline validation fails

Run:

```powershell
$env:PYTHONPATH='lca_vessel_tree_generator'
.\.venv\Scripts\python.exe -m LCA_topology_generator.controlled_anatomical_lca_synthetic --input-dir outputs\dataset_lca_anatomical --output-dir outputs\dataset_lca_anatomical_synthetic --metadata-dir lca_vessel_tree_generator\LCA_branch_control_points\generated
```

Synthetic control-point outputs are written under:

```text
outputs/dataset_lca_anatomical_synthetic/trees/patient_XXXX_synth_YYY/
```

The synthetic full pipeline output is written under:

```text
outputs/dataset_lca_anatomical_synthetic/pipeline/
```

The generated summary is:

```text
outputs/dataset_lca_anatomical_synthetic/SYNTHETIC_LCA_GENERATION_SUMMARY.md
```

This synthetic set is patient-inspired and anatomically constrained. It is still
not clinical patient-specific reconstruction.
