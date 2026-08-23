# Coronary4D Final Production Presentation

This directory contains only the md-aligned surface-relative production generator output. It is an LCA-only research/engineering prototype and is not clinically validated.

## ParaView

- Open `final_static_tree.vtm` for the unchanged canonical LMCA/LAD/LCX centerlines.
- Open `final_tree_with_scaffold.vtm` to add the exact associated production ellipsoid.
- Open `cine.pvd`, click Apply, and press Play for the validated closed 4D cycle.
- Use Tube representation for vessels. Color the cine blocks by `radius_mm`; `disease_reduction_fraction` is present and is zero for this healthy presentation case.

The canonical case is `tree_0015`, selected previously as a warning-free production medoid. No presentation coordinate translation or geometry deformation is applied.
