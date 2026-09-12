# 4D coronary LCA case

Case: `diffuse_lcx`. Validation: **PASS**.

Open `vtk/cine.pvd` in ParaView to animate the cardiac phases. The NumPy tensor `geometry_cine.npy` uses shape `(phase, branch, point, x/y/z/radius)` with branch order LMCA, LAD, LCX. Disease parameters and deterministic provenance are in `metadata.json`. From inside this exported case directory, run `python -m vessel_tree_generator visualize --input-dir . --clean` to create dashboards, GIF, and interactive HTML. This is a research/engineering prototype, not a clinically validated model.
