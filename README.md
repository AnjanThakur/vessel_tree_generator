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
|   |-- outputs/
|   |   |-- dataset_lca/
|   |   |-- synthetic_lca/
|-- rca_vessel_tree_generator/
|   |-- README.md
|   |-- requirements.txt
|   |-- RCA_branch_control_points/
|   |-- tube_generator.py
|   |-- tube_functions.py
|   |-- fwd_projection_functions.py
```

## LCA Workflows

Dataset-derived LCA generation:

```bash
python -m lca_vessel_tree_generator.LCA_topology_generator.generate_lca_dataset
```

Experimental synthetic sampling:

```bash
python -m lca_vessel_tree_generator.LCA_topology_generator.generate_lca
```

## Requirements Overview

- **LCA**: Requires `numpy`, `matplotlib`, and `geomdl`.
- **RCA**: Requires `numpy`, `matplotlib`, `scikit-image`, and `geomdl`.

Please see the respective subproject `requirements.txt` files for installation details.
