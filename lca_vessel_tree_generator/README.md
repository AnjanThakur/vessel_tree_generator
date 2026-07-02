# LCA Patient Centerline Preprocessor

This module processes clinical/patient coronary centerline data, normalizes their coordinate systems, and extracts control points.

The primary output is the generation of control-point database `.npy` files for the population of patients.

## Workflow

1. Parses raw clinical paths/contours from `LCA_dataset/` (supports `.pth`, `.ctgr`, and `.vtp` formats).
2. Translates the ostium to `(0, 0, 0)`.
3. Aligns the LMCA trunk direction along the `+X` axis.
4. Rotates around the `X` axis to align the LMCA bifurcation plane (LMCA -> LAD + LCX) normal to the `+Z` axis.
5. Resamples each branch into fixed numbers of control points:
   - **LMCA**: 5 control points
   - **LAD**: 12 control points
   - **LCX**: 10 control points
6. Saves individual patient normalized centerlines under `processed_dataset/`.
7. Exports collective patient control point matrices and unified tree statistics to `LCA_branch_control_points/generated/`.

## Generated Files

Running the preprocessor writes clinical databases to `LCA_branch_control_points/generated/`:

- `LMCA_patient_ctrl_points.npy` (shape `(M, 5, 3)`)
- `LAD_patient_ctrl_points.npy` (shape `(M, 12, 3)`)
- `LCX_patient_ctrl_points.npy` (shape `(M, 10, 3)`)

And the unified/connected LCA tree datasets:
- `LCA_tree_ctrl_points.npy` (shape `(M, 27, 3)`) - Stacked patient trees
- `LCA_tree_mean.npy` (shape `(27, 3)`) - Coordinate-wise mean shape
- `LCA_tree_std.npy` (shape `(27, 3)`) - Coordinate-wise standard deviation (variation)

Where `M` is the number of successfully preprocessed patient cases, and `3` stores x/y/z coordinates in millimeters.

### Strict Indexing Alignment
The 27 points in the unified tree represent a strict topological concatenation order that must never be altered:
- **Indices `0` to `4`**: LMCA control points (5 points)
- **Indices `5` to `16`**: LAD control points (12 points)
- **Indices `17` to `26`**: LCX control points (10 points)

Additionally, a quality control report is written to:
- `processed_dataset/preprocessing_qc_report.json`

An overlay plot of all normalized patient centerlines is saved to:
- `processed_dataset/all_normalized_centerlines.png`

## Running Preprocessing

From the repository root:

```bash
python -m LCA_topology_generator.patient_preprocessor
```
