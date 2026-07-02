# Coronary Vessel Tree Generators

This repository contains two decoupled subprojects for generating 3D coronary artery trees and centerlines:

1. **Left Coronary Artery (LCA) Pipeline**: [lca_vessel_tree_generator](./lca_vessel_tree_generator/)
2. **Right Coronary Artery (RCA) & Primitive Vessel Pipeline**: [rca_vessel_tree_generator](./rca_vessel_tree_generator/)

For detailed guides, dependencies, and execution instructions, please refer to the respective subproject READMEs.

---

## Directory Overview

```
latest_lca/
├── README.md                          <-- This file (repository overview)
│
├── lca_vessel_tree_generator/         <-- LCA centerline generator and patient preprocessor
│   ├── README.md                      <-- LCA documentation & usage instructions
│   ├── requirements.txt               <-- Python dependencies for LCA
│   ├── LCA_topology_generator/        <-- Python source package (preprocessor, sampler, B-spline, visualize)
│   ├── LCA_branch_control_points/     <-- Input control points databases (clinical & generated)
│   ├── processed_dataset/             <-- Preprocessing patient centerline alignment outputs
│   └── outputs/                       <-- Target output folder for generated synthetic centerlines and plots
│
└── rca_vessel_tree_generator/         <-- RCA, spline, and cylinder 3D surface/projection generator
    ├── README.md                      <-- RCA documentation & usage instructions
    ├── requirements.txt               <-- Python dependencies for RCA
    ├── RCA_branch_control_points/     <-- RCA reference control points
    ├── outputs/                       <-- Target folder for generated 3D surface labels and 2D projections
    ├── example_images/                <-- Sample visual outputs
    ├── tube_generator.py              <-- Main execution script
    ├── tube_functions.py              <-- 3D tube geometry generation math
    └── fwd_projection_functions.py    <-- Forward projections and skimage processing
```

---

## Requirements Overview

- **LCA**: Requires `numpy`, `matplotlib`, and `geomdl`.
- **RCA**: Requires `numpy`, `matplotlib`, `scikit-image`, and `geomdl`.

Please see the respective subproject `requirements.txt` files for installation details.
