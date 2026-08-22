# LCA synthetic-tree realization

Open `vtk/synthetic_tree.vtm` in ParaView.

Generation mode: `frozen_lca_statistics`. This tree was sampled from the frozen LCA ellipsoid, landmark, empirical trajectory, PCA, and validation package. The generator consumes ellipsoid surface functions, generates smooth B-spline paths, enforces `LMCA[-1] == LAD[0] == LCX[0]`, validates the result, and exports modular VTK.

Validation status: `PASS`. Population statistics and PCA are used only when a complete frozen LCA package is explicitly supplied.
