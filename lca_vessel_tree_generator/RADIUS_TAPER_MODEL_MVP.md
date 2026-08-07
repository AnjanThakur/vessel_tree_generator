# Radius Taper Model MVP

## Why Fixed Distal Percentages Were Replaced

The earlier static radius MVP used fixed distal percentages:

- LMCA distal radius = 85% of LMCA proximal radius
- LAD distal radius = 55% of LAD proximal radius
- LCX distal radius = 55% of LCX proximal radius

That was useful for a first working pipeline, but it made tapering independent of
branch length and vessel role. A short branch and a long branch could end with
the same distal percentage, even though their physical distances are different.

The updated MVP replaces fixed distal percentages with a rule-based model that
uses cumulative branch distance, branch type, and a simple cube-law bifurcation
consistency check.

## Distance-Based Taper

For distributing branches such as LAD and LCX, radius is calculated with:

```text
radius(distance) = proximal_radius * exp(-k * distance)
```

Where:

- `distance` is cumulative arc length along the branch in millimeters
- `proximal_radius` comes from patient metadata or MVP defaults
- `k` is a configurable taper rate

This means longer branches taper more than shorter branches, and users can tune
the taper rate without changing the rest of the pipeline.

## Branch Types

The current LCA MVP uses:

- `LMCA`: `parent_trunk`
- `LAD`: `distributing`
- `LCX`: `distributing`

Future branch types can use the same framework:

- `RCA`: `distributing`
- diagonal, OM, septal, marginal: `delivering`

Default taper rates are configurable in `radius_model.py`:

```text
parent_trunk: 0.0025 per mm
distributing: 0.0035 per mm
delivering: 0.0065 per mm
```

Smaller delivering vessels can therefore taper faster than main distributing
vessels.

## LMCA Cube-Law Bifurcation

LMCA is handled as a parent trunk. Its distal radius is estimated from LAD and
LCX proximal radii using a Murray/Zamir-style cube-law relation:

```text
parent_radius^3 ~= child1_radius^3 + child2_radius^3

lmca_distal = (lad_proximal^3 + lcx_proximal^3)^(1/3)
```

LMCA then tapers from its proximal radius to this cube-law distal radius using a
smooth distance-shaped transition.

Validation checks that the generated LMCA distal radius remains compatible with
the LAD/LCX proximal radii within a configurable tolerance.

## Output Format

The output format is unchanged:

```text
[x, y, z, radius]
```

Existing downstream modules can reuse the improved radius outputs:

- disease / stenosis
- plaque surface deformation
- tube surface generation
- tight mesh generation

## Validation Checks

The radius validation checks:

- radius values are positive
- no NaN or Inf values are present
- adjacent radius changes are smooth
- proximal radius is greater than or equal to distal radius
- LMCA proximal radius is larger than LAD/LCX proximal radius
- LMCA distal radius is compatible with LAD/LCX proximal radii by cube-law tolerance

## Current Limitations

- This is still an MVP rule-based model, not a clinically calibrated radius model.
- It does not infer patient-specific mid/distal radius from imaging contours.
- It does not model disease, plaque, pulsatility, flow, or cardiac motion.
- LMCA cube-law consistency uses LAD/LCX proximal radii only.
- Branch type labels remain configurable numeric rules, not fixed clinical classes.

## Future Calibration

Future improvements can calibrate taper rates using patient-specific:

- proximal radius
- mid-branch radius
- distal radius
- branch length
- vessel role and downstream territory

That would allow the same modular taper framework to become more data-driven
without changing the `[x, y, z, radius]` interface.
