# LCA Patient Centerline Preprocessor

This module processes clinical/patient coronary centerline data, normalizes their coordinate systems, and extracts control points.

The primary workflow is dataset-driven: patient LCA centerlines are normalized, converted into fixed-topology control-point trees, interpolated into smooth centerlines, and measured for tortuosity.

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
8. Generates dataset-derived centerlines, tortuosity metrics, and vessel-style visualizations under `outputs/dataset_lca/`.

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

The QC report includes per-branch tortuosity metrics for LMCA, LAD, and LCX:
- `path_length`: cumulative 3D centerline length
- `chord_length`: straight-line endpoint distance
- `tortuosity`: `path_length / chord_length`

Dataset-derived generation writes to `outputs/dataset_lca/`:
- `trees/patient_XXXX/control_points_27x3.npy`
- `trees/patient_XXXX/lmca_centerline.npy`
- `trees/patient_XXXX/lad_centerline.npy`
- `trees/patient_XXXX/lcx_centerline.npy`
- `trees/patient_XXXX/tortuosity_metrics.json`
- `trees/patient_XXXX/tree_visualization.png`
- `trees/patient_XXXX/tree_3d_matrix.png`
- `trees/patient_XXXX/controlled_tortuosity_variants.png`
- `trees/patient_XXXX/controlled_tortuosity_variants/`
- `01_dataset_trees.png`
- `01_dataset_trees_3d_matrix.png`
- `02_dataset_tree_with_controlled_tortuosity.png`
- `03_branch_controlled_tortuosity_examples.png`
- `04_dataset_trees_with_measured_tortuosity.png`
- `05_dataset_branch_tortuosity_ranking.png`
- `validation_report.json`
- `summary.json`

Synthetic experimental outputs are kept separate under `outputs/synthetic_lca/`.

## Running Preprocessing

From the repository root:

```bash
python -m lca_vessel_tree_generator.LCA_topology_generator.patient_preprocessor
```

## Running Dataset-Derived LCA Generation

From the repository root:

```bash
python -m lca_vessel_tree_generator.LCA_topology_generator.generate_lca_dataset
```

## Running Experimental Synthetic Sampling

This path samples each control-point coordinate from population mean/std and is kept separate because it can produce anatomically poor trees without validation:

```bash
python -m lca_vessel_tree_generator.LCA_topology_generator.generate_lca
```
