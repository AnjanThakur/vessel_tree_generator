# Disease / Stenosis Module MVP

## What This Module Does

The disease module adds simple static stenosis to existing dataset-driven LCA
trees. It takes radius-enabled centerlines and reduces the radius along selected
branch segments.

The module is geometry-independent. It does not assume a specific LAD or LCX
orientation, and it can be rerun after corrected anatomical LCA geometry is
added later.

## Input Format

Each branch centerline uses:

```text
[x, y, z, radius]
```

Supported normal inputs include:

- `tree_centerline_radius.npz`
- `lmca_centerline_radius.npy`
- `lad_centerline_radius.npy`
- `lcx_centerline_radius.npy`

The disease module modifies only the radius column. The `x`, `y`, and `z`
coordinates are preserved exactly.

## Modular Disease Architecture

The disease pipeline is split into two layers:

```text
Layer 1: radius-level stenosis
- focal
- diffuse
- tandem
- works on [x, y, z, radius]

Layer 2: optional plaque surface deformation
- symmetric plaque
- eccentric plaque
- modifies tube cross-section rings
```

The current radius-level disease module remains valid. Optional plaque is an
advanced mesh/surface feature and does not replace the radius outputs.

## Supported Disease Types

### Focal Stenosis

A short local narrowing around a normalized center position.

```json
{
  "case_id": "patient_0001_lad_focal_70",
  "lesions": [
    {
      "branch": "LAD",
      "type": "focal",
      "center": 0.45,
      "length": 0.10,
      "severity": 0.70
    }
  ]
}
```

### Diffuse Stenosis

A longer gradual narrowing over a normalized branch segment.

```json
{
  "case_id": "patient_0001_lcx_diffuse_40",
  "lesions": [
    {
      "branch": "LCX",
      "type": "diffuse",
      "start": 0.30,
      "end": 0.70,
      "severity": 0.40
    }
  ]
}
```

### Tandem Stenosis

Multiple lesions on the same branch. The MVP examples represent tandem disease
as multiple focal lesions.

```json
{
  "case_id": "patient_0001_lad_tandem",
  "lesions": [
    {
      "branch": "LAD",
      "type": "focal",
      "center": 0.35,
      "length": 0.08,
      "severity": 0.50
    },
    {
      "branch": "LAD",
      "type": "focal",
      "center": 0.65,
      "length": 0.10,
      "severity": 0.70
    }
  ]
}
```

### Optional Eccentric Plaque

Eccentric plaque can be requested per lesion:

```json
{
  "case_id": "patient_0001_lad_focal_70_eccentric",
  "lesions": [
    {
      "branch": "LAD",
      "type": "focal",
      "center": 0.45,
      "length": 0.10,
      "severity": 0.70,
      "plaque_mode": "eccentric",
      "plaque_angle": 90,
      "eccentricity": 0.75
    }
  ]
}
```

Meaning:

- `severity`: maximum radius narrowing
- `center`: normalized lesion location
- `length`: normalized lesion length
- `plaque_mode`: `symmetric` or `eccentric`
- `plaque_angle`: side of the generated tube ring where plaque effect is strongest
- `eccentricity`: how one-sided the plaque is, from `0` to `1`

For eccentric plaque, the radius-level diseased centerline still stores the
maximum luminal narrowing. The optional plaque surface redistributes that
narrowing around each tube ring so one side is narrowed more than the opposite
side.

## Meaning Of Severity

Severity is percentage radius reduction at maximum narrowing.

Example:

```text
severity = 0.70
new_radius = original_radius * (1 - 0.70)
```

So a 70% stenosis leaves 30% of the original radius at the lesion center, unless
the configurable minimum radius clamp is reached.

## How To Run

From `lca_vessel_tree_generator`:

```powershell
..\.venv\Scripts\python.exe -m LCA_topology_generator.apply_disease_to_lca
```

By default, this reads:

```text
outputs/dataset_lca/trees
```

and writes:

```text
outputs/dataset_lca_disease
```

To run one custom config:

```powershell
..\.venv\Scripts\python.exe -m LCA_topology_generator.apply_disease_to_lca --config path\to\disease_config.json
```

To skip diseased tube and mesh regeneration:

```powershell
..\.venv\Scripts\python.exe -m LCA_topology_generator.apply_disease_to_lca --skip-mesh
```

## Output Files

Each patient/case folder contains:

- `tree_centerline_radius_diseased.npz`
- `lmca_centerline_radius_diseased.npy`
- `lad_centerline_radius_diseased.npy`
- `lcx_centerline_radius_diseased.npy`
- `disease_config.json`
- `disease_validation.json`
- `disease_summary.json`
- `diseased_radius_profile.png`
- `diseased_tree_with_radius.png`

When tube and mesh generation is enabled, each case also contains:

- `diseased_tube_surface.npz`
- `diseased_tight_mesh.npz`
- `diseased_tight_mesh.ply`
- `diseased_tight_mesh.stl`
- `diseased_tight_mesh.png`
- `diseased_tight_mesh_validation.json`

If a lesion uses `"plaque_mode": "eccentric"`, the case also contains:

- `plaque_tube_surface.npz`
- `plaque_tight_mesh.npz`
- `plaque_tight_mesh.ply`
- `plaque_tight_mesh.stl`
- `plaque_tight_mesh.png`
- `plaque_surface_validation.json`
- `plaque_tight_mesh_validation.json`
- `plaque_summary.json`

Root-level summary files are saved in `outputs/dataset_lca_disease`:

- `disease_summary.json`
- `disease_validation.json`

## Validation Checks

The MVP validation checks:

- branch name is one of `LMCA`, `LAD`, or `LCX`
- lesion type is supported
- severity is `>= 0` and `< 1`
- lesion positions are normalized from `0` to `1`
- focal lesion length is positive and stays inside branch range
- diffuse lesion `end` is greater than `start`
- radius values remain positive
- no NaN or Inf values are present
- diseased radius does not increase above baseline radius
- diseased radius respects the minimum radius clamp
- `x`, `y`, and `z` coordinates are unchanged
- adjacent radius jumps are reported as warnings when above threshold
- plaque mode is `symmetric` or `eccentric`
- eccentricity is normalized from `0` to `1`
- optional plaque surfaces are finite and match branch centerline length

## Current MVP Limitations

- Radius-level disease modifies radius only.
- Optional eccentric plaque deforms tube surface rings only.
- Eccentric plaque angle is defined in the generated tube ring frame.
- It does not model plaque material.
- It does not model blood flow.
- It does not add cardiac motion.
- It does not add pulsatility.
- It does not perform clinical stenosis validation.
- Diseased tube and tight mesh generation reuse the current MVP surface and mesh
  modules and are not CFD-grade or clinical-grade.
- The module is geometry-independent and should be rerun after corrected LCA
  anatomy is ready.
