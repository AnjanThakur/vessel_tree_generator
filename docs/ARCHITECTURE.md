# Coronary4D architecture

## Production data flow

```text
Frozen 52-case LCA statistics
        |
        v
case-matched baseline + 81-D surface-deviation PCA
        |
        v
LMCA/LAD/LCX anatomy gates
        |
        +--> parametric radius taper
        +--> controlled radius-only disease
        +--> ellipsoid-relative cardiac motion
        +--> phase-dependent pulsatility/compliance
        |
        v
NumPy + JSON + VTK/PVD export and independent readback
```

## Runtime packages

- `vessel_tree_generator`: supported API, CLI, disease, radius, motion,
  validation, export, visualization, audit and surface-mesh utilities.
- `pca_ssm_vessel_tree_generator`: population representation, PCA sampling,
  anatomical gates, B-spline generation and cardiac-motion implementation.
- `outputs/lca_ssm/lca_population_model/generator_statistics`: compact frozen
  statistics required by `CoronaryTreeGenerator`.

The runtime does not require raw NIfTI data, extraction intermediates, old RCA
utilities or historical two-ellipse experiments. Recomputing the statistical
model does require the protected source dataset, which is intentionally not
distributed.

## Public contract

The supported entry point is:

```python
from vessel_tree_generator import CoronaryTreeGenerator, stenosis_config
```

The supported CLI entry point is:

```text
python -m vessel_tree_generator
```

Generated anatomy is LCA-only: LMCA, LAD and LCX. Disease modifies radii, not
centerline coordinates. Radius taper, disease, motion, pulsatility and lesion
compliance are explicit parametric research models rather than learned patient
distributions.

## Artifact tiers

1. `submission_release/final_presentation/` — compact canonical static and 4D
   demonstration with fixed preview images.
2. `submission_release/final_presentation_52/` — all 52 accepted trees, each
   with centerlines, scaffold, closed cine, healthy tapered mesh and diseased
   tapered mesh.
3. `submission_release/final_audit/` and `final_validation/` — machine-readable
   population, PCA, novelty, holdout, VTK and integrity evidence.
4. `submission_release/presentation_center/` — local presentation/verification
   application and projector-ready assets.

## Integrity boundaries

- Frozen statistics are SHA-256 locked by
  `generator_statistics_manifest.json`.
- Generated presentation meshes are visualization copies; their manifest
  records zero source-coordinate change.
- Every exported 4D case preserves branch point correspondence and exact
  LMCA-to-LAD/LCX junctions at every phase.
- Engineering validation does not constitute clinical validation.
