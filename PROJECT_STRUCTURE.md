# Project Structure

This repository keeps source code, input control-point data, and generated
outputs separate.

```text
vessel_tree_generator/
|-- README.md
|-- PROJECT_STRUCTURE.md
|-- outputs/
|   |-- dataset_lca_anatomical/
|   |-- dataset_lca_anatomical_pipeline/
|   |-- dataset_lca_anatomical_synthetic/
|   |-- legacy_lca_outputs/
|-- lca_vessel_tree_generator/
|   |-- LCA_branch_control_points/
|   |   |-- generated/
|   |-- LCA_topology_generator/
|   |   |-- paths.py
|   |   |-- heart_landmark_model.py
|   |   |-- anatomical_lca_reorientation.py
|   |   |-- anatomical_lca_pipeline.py
|   |   |-- controlled_anatomical_lca_synthetic.py
|   |   |-- generate_lca_dataset.py
|   |   |-- apply_disease_to_lca.py
|   |   |-- radius_model.py
|   |   |-- disease_model.py
|   |   |-- tube_surface.py
|   |   |-- tight_mesh.py
|-- rca_vessel_tree_generator/
```

## Output Rule

All generated outputs should go under the repository-root `outputs/` folder.
The old `lca_vessel_tree_generator/outputs/` location has been moved into:

```text
outputs/legacy_lca_outputs/
```

Current scripts use `LCA_topology_generator/paths.py` to resolve canonical
paths, so running from the repo root or package module should not recreate a
second output folder.

## Current LCA Output Layers

```text
outputs/dataset_lca_anatomical/
```

Corrected anatomical 27x3 LCA control points and VTK/PNG previews.

```text
outputs/dataset_lca_anatomical_pipeline/
```

Real/anatomically corrected trees passed through centerline, radius, disease,
tube surface, and tight mesh generation.

```text
outputs/dataset_lca_anatomical_synthetic/
```

Controlled synthetic anatomical variants plus their full downstream pipeline.

```text
outputs/legacy_lca_outputs/
```

Older generated results preserved for reference. These are not the canonical
current outputs.
