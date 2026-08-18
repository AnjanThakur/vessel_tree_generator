# Person 2 synthetic-tree demo

Open `vtk/synthetic_tree.vtm` in ParaView.

Generation mode: `person1_frozen_statistics`. This tree was sampled from the frozen Person 1 ellipsoid, landmark, empirical trajectory, PCA, and validation package. The generator consumes ellipsoid surface functions, generates smooth B-spline paths, enforces `LMCA[-1] == LAD[0] == LCX[0]`, validates the result, and exports modular VTK.

Validation status: `PASS`. Population statistics and PCA are used only when a complete frozen Person 1 package is explicitly supplied.
