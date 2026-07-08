# Coronary Vessel Tree Generators

This repository contains two decoupled subprojects for generating 3D coronary artery trees and centerlines:

1. **Left Coronary Artery (LCA) Pipeline**: [lca_vessel_tree_generator](./lca_vessel_tree_generator/)
2. **Right Coronary Artery (RCA) & Primitive Vessel Pipeline**: [rca_vessel_tree_generator](./rca_vessel_tree_generator/)

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
|-- outputs/
|   |-- dataset_lca_anatomical/
|   |-- dataset_lca_anatomical_pipeline/
|   |-- dataset_lca_anatomical_synthetic/
|   |-- legacy_lca_outputs/
```

## LCA Workflows

Dataset-derived LCA generation:

```bash
python -m lca_vessel_tree_generator.LCA_topology_generator.generate_lca_dataset
```

This exports validated patient-derived LMCA/LAD/LCX centerlines, measured tortuosity, static radius taper profiles, simple tube surfaces, MVP connected tight meshes, and visual checks under root `outputs/dataset_lca/`. Radius-enriched LCA arrays use `N x 4` branch format `[x, y, z, radius_mm]`; tube surfaces use `N x circle_points x 3`; tight meshes use `vertices` (`V x 3`) and triangular `faces` (`F x 3`).

Experimental synthetic sampling:

```bash
python -m lca_vessel_tree_generator.LCA_topology_generator.generate_lca
```

Generated files should live under the repository-root `outputs/` folder. The package-level `lca_vessel_tree_generator/outputs/` folder is legacy and should not be recreated by current scripts.

## Requirements Overview

- **LCA**: Requires `numpy`, `matplotlib`, and `geomdl`.
- **RCA**: Requires `numpy`, `matplotlib`, `scikit-image`, and `geomdl`.

Please see the respective subproject `requirements.txt` files for installation details.
