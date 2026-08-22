# Mentor Visualization Pack

This folder is a self-contained ParaView presentation package for the verified
major-vessel LCA research generator. Keep the directory structure unchanged:
the `.pvd` and `.vtm` index files use relative paths to their `.vtp` data.

## Start here in ParaView

1. Open `01_static_population/OPEN_ALL_52_TREES.vtm` to show the full generated
   population. In the Pipeline Browser, expand individual tree blocks. Hide the
   ellipsoid blocks when you want an uncluttered coronary-only comparison.
2. Open `02_4d_disease_cases/healthy/OPEN_4D_CINE.pvd`, click **Apply**, then
   press **Play** to demonstrate the closed cardiac cycle.
3. Repeat with `focal_lad`, `diffuse_lcx`, and `tandem_lad`. These four cases use
   the same representative anatomy, so differences isolate the disease model.
4. For disease visualization, select the vessel blocks and color by
   `disease_reduction_fraction`. For calibre and pulsatility, color by
   `radius_mm` and play the time series.

Recommended display settings: use **Tube** representation with radius about
`0.5-0.8 mm`, keep the scalar bar visible, use a white background, and use
**Reset Camera** after opening each dataset.

## What each entry demonstrates

| Entry | Main point |
|---|---|
| `OPEN_ALL_52_TREES.vtm` | Population diversity, ellipsoid scaffold, LMCA/LAD/LCX identity, landmarks and surface-relative arrays |
| `healthy/OPEN_4D_CINE.pvd` | Cardiac motion and global/local pulsatility without disease |
| `focal_lad/OPEN_4D_CINE.pvd` | One localized LAD stenosis with reduced lesion compliance |
| `diffuse_lcx/OPEN_4D_CINE.pvd` | Extended LCX disease region |
| `tandem_lad/OPEN_4D_CINE.pvd` | Multiple separated LAD stenoses |

Static population branches expose `surface_u_rad`, `surface_v_rad`,
`surface_normal_offset_mm` and local deviation coefficients. The 4D case
branches expose `radius_mm`, `normalized_arc_length` and
`disease_reduction_fraction`.

## Suggested mentor explanation

- The two fitted ellipses measure the triaxial ellipsoid scaffold; they are not
  LAD or LCX vessel molds.
- LAD mainly progresses base-to-apex in surface `v`; LCX mainly travels
  circumferentially in surface `u`.
- The 81-D PCA models 27 surface-relative points with tangent-u, tangent-v and
  normal coefficients across LMCA, LAD and LCX.
- Disease changes radius, not centerline geometry.
- Cardiac deformation moves the ellipsoid-associated vessels coherently, while
  radius pulsatility and reduced lesion compliance remain visible over time.
- Scope remains LCA-only and research/engineering; it is not clinically
  validated.

The files in `03_presentation_assets` are quick-reference images and GIFs for a
meeting where ParaView is unavailable. `VISUALIZATION_PACK_MANIFEST.json`
contains hashes and machine verification results.
