# Coronary Vessel Tree Generators

This repository contains three decoupled subprojects for extracting and generating 3D coronary artery trees and centerlines:

1. **Left Coronary Artery (LCA) Pipeline**: [lca_vessel_tree_generator](./lca_vessel_tree_generator/)
2. **Right Coronary Artery (RCA) & Primitive Vessel Pipeline**: [rca_vessel_tree_generator](./rca_vessel_tree_generator/)
3. **PCA/SSM Landmark Extraction Pipeline**: [pca_ssm_vessel_tree_generator](./pca_ssm_vessel_tree_generator/)

The current LCA work is dataset-driven. Patient-derived LCA control trees are stored separately from experimental synthetic sampling outputs.

## Directory Overview

```text
vessel_tree_generator/
|-- README.md
|-- lca_vessel_tree_generator/
|   |-- README.md
|   |-- requirements.txt
|   |-- LCA_topology_generator/
|   |   |-- patient_preprocessor.py
|   |   |-- generate_lca_dataset.py
|   |   |-- generate_lca.py
|   |   |-- radius_model.py
|   |   |-- tortuosity.py
|   |   |-- bspline.py
|   |   |-- visualize.py
|   |-- LCA_branch_control_points/
|   |   |-- generated/
|   |   |   |-- LCA_tree_ctrl_points.npy
|   |   |   |-- LCA_tree_mean.npy
|   |   |   |-- LCA_tree_std.npy
|   |   |   |-- LMCA_patient_ctrl_points.npy
|   |   |   |-- LAD_patient_ctrl_points.npy
|   |   |   |-- LCX_patient_ctrl_points.npy
|-- rca_vessel_tree_generator/
|   |-- README.md
|   |-- requirements.txt
|   |-- RCA_branch_control_points/
|   |-- tube_generator.py
|   |-- tube_functions.py
|   |-- fwd_projection_functions.py
|-- pca_ssm_vessel_tree_generator/
|   |-- README.md
|   |-- requirements.txt
|   |-- 01_extract_landmarks.py
|   |-- 02_landmark_selection.py
|   |-- build_lca_ssm_population_stats.py
|   |-- lca_ssm_planes.py
|   |-- lca_ssm_curve_fitting.py
|   |-- LCA_SSM_PLAN_AFTER_LANDMARKS.md
|   |-- centerlines/
|   |-- nii files/
|-- outputs/
|   |-- generated LCA datasets and visualizations
```

## LCA Workflows

Dataset-derived LCA generation:

```bash
python -m lca_vessel_tree_generator.LCA_topology_generator.generate_lca_dataset
```

This exports validated patient-derived LMCA/LAD/LCX centerlines, measured tortuosity, static radius taper profiles, simple tube surfaces, MVP connected tight meshes, and visual checks under repository-root `outputs/dataset_lca/`. Radius-enriched LCA arrays use `N x 4` branch format `[x, y, z, radius_mm]`; tube surfaces use `N x circle_points x 3`; tight meshes use `vertices` (`V x 3`) and triangular `faces` (`F x 3`). Radius tapering is distance-based and branch-type-based, with an LMCA cube-law bifurcation relation.

Experimental synthetic sampling:

```bash
python -m lca_vessel_tree_generator.LCA_topology_generator.generate_lca
```

## PCA/SSM Workflow

Extract skeleton landmarks and paths from the configured NIfTI segmentation:

```bash
python pca_ssm_vessel_tree_generator/01_extract_landmarks.py
```

Select the LMCA bifurcation and two dominant downstream branches:

```bash
python pca_ssm_vessel_tree_generator/02_landmark_selection.py
```

Both stages open Napari for interactive 3D inspection. See the subproject README for data assumptions and generated files.

Fit post-landmark LCA planes, curves, per-patient parameters, and population statistics from the 200 PCA label volumes and centerline graphs:

```bash
python pca_ssm_vessel_tree_generator/build_lca_ssm_population_stats.py --clean
```

This non-interactive stage writes only to `outputs/lca_ssm/`. It reproduces the existing root/bifurcation selection and infers LAD/LCX candidate identities from the cohort's common NIfTI RAS orientation; assignment confidence and warnings are retained for review.

## Requirements Overview

- **LCA**: Requires `numpy`, `matplotlib`, `geomdl`, and `scipy` (for VMTK extraction and validation).
- **RCA**: Requires `numpy`, `matplotlib`, `scikit-image`, and `geomdl`.
- **PCA/SSM**: Requires `numpy`, `nibabel`, `napari`, `scikit-image`, `skan`, `scipy`, `matplotlib`, `pandas`, and `networkx`.

Please see the respective subproject `requirements.txt` files for installation details.
