# ParaView Demo Guide - Focal LAD Case

## Exact entry point

The application resolves and displays this path at runtime:

```text
C:\Internship\vessel_tree_generator\submission_release\demo_cases\focal_lad\vtk\cine.pvd
```

The portable relative path is:

```text
submission_release\demo_cases\focal_lad\vtk\cine.pvd
```

## Demonstration sequence

1. In ParaView, choose **File > Open** and select `cine.pvd`.
   - **SAY:** “PVD is the single time-series entry point.”
2. Click **Apply** in the Properties panel.
   - **SAY:** “Each time step references one VTM multiblock coronary tree.”
3. Expand the multiblock hierarchy if needed.
   - **SAY:** “The blocks preserve LMCA, LAD and LCX identity.”
4. Set **Representation** to a tube-compatible view if desired, or apply the
   **Tube** filter when the reader requires line thickening.
   - **SAY:** “The underlying exported objects are centerlines with point data.”
5. Use **Color By > radius_mm**.
   - **SAY:** “Radius is stored at every time-corresponded vessel point.”
6. Press **Play** in the animation toolbar.
   - **SAY:** “Ten frames are stored: nine independent cardiac positions and a
     repeated closure frame.”
7. Pause near peak contraction and inspect the LAD.
   - **SAY:** “The focal LAD narrowing changes radius, while XYZ anatomy remains
     identical to the healthy reference before cardiac deformation.”
8. Change coloring to **disease_reduction_fraction**.
   - **SAY:** “This field isolates lesion severity from the global radius field.”
9. Toggle LMCA, LAD and LCX blocks.
   - **SAY:** “LMCA ends exactly where both daughters start in every phase.”

## What the files mean

```text
cine.pvd
  -> phase_000/tree.vtm ... phase_009/tree.vtm
      -> LMCA / LAD / LCX VTP blocks
```

The final release audit reports VTK readback **PASS** for all four demonstration
cases. ParaView is a visualization client; it is not required for numerical
verification because the exported files are also read back programmatically.
