# PCA/SSM Vessel Tree Landmark Extraction

This subproject extracts coronary skeleton landmarks, selects an LMCA trunk plus two dominant downstream branches from segmented NIfTI volumes, and fits post-landmark LCA statistical-shape parameters across the supplied 200-case PCA label cohort.

It was imported from [Gauravch-dev/pca-ssm_vessel_tree_generator](https://github.com/Gauravch-dev/pca-ssm_vessel_tree_generator) at commit `acc98bc`. The upstream Git metadata is intentionally not nested inside this repository.

## Data

- `nii files/` contains the input `.nii.gz` label volumes.
- `centerlines/` contains the corresponding centerline data supplied upstream.
- The root `.npy`, `.npz`, and `.csv` files are the sample intermediate and selected-branch outputs supplied upstream.

The default input configured in `01_extract_landmarks.py` is `nii files/117.label.nii.gz`.

## Workflow

From the parent repository root, run:

```bash
python pca_ssm_vessel_tree_generator/01_extract_landmarks.py
python pca_ssm_vessel_tree_generator/02_landmark_selection.py
```

Stage 1 skeletonizes the segmentation, calculates endpoint radii, chooses a candidate root, and writes `landmarks.npy`, `summary.csv`, and `branches.npz`.

Stage 2 builds a shortest-path tree, selects an LMCA bifurcation, and writes `LMCA.npy`, `BranchA.npy`, `BranchB.npy`, their node arrays, and `selected_landmarks.npy`.

Both scripts open Napari for interactive 3D inspection. Branch A and Branch B are selected by length-based rules; they are not automatically classified as LAD versus LCX.

## Post-landmark LCA SSM stage

The non-interactive SSM stage reads all matching `centerlines/<case>/summary.csv`, `branches.npz`, and `nii files/<case>.nii.gz` inputs. It reproduces the existing largest-endpoint-radius root selection and stage-2 LMCA bifurcation thresholds. Because the source does not annotate the daughter names, it classifies the selected branches as LAD/LCX candidates from the shared NIfTI RAS orientation and records the assignment rule, score, margin, and review warnings in every patient JSON.

Run it from the repository root:

```bash
python pca_ssm_vessel_tree_generator/build_lca_ssm_population_stats.py --clean
```

It fits SVD planes, an LCX ellipse arc with a PCA/quadratic fallback, an LAD guide curve, per-patient anatomical parameters, cohort statistics, a generator schema, rejection records, and visual checks under `outputs/lca_ssm/`. In the current audit, 191 of 200 cases pass and 9 fail the unchanged bifurcation thresholds. See [LCA_SSM_PLAN_AFTER_LANDMARKS.md](./LCA_SSM_PLAN_AFTER_LANDMARKS.md) for the data contract, methods, validation rules, and limitations.

## Isolated two-ellipse prototype

Before changing the population model, the joined LAD/LCX ellipse approach can be reviewed on `1.label` alone:

```bash
python pca_ssm_vessel_tree_generator/prototype_1_label_heart_scaffold.py --clean
```

This prototype constrains both planes to the shared bifurcation and corresponding terminal, uses the LMCA bifurcation tangent to orient both right-handed plane bases, and fits endpoint-exact LAD and LCX ellipse intervals. It writes only to `outputs/lca_ssm/prototype_1_label/`; it does not update population statistics or the generator schema.

## Installation

```bash
python -m pip install -r pca_ssm_vessel_tree_generator/requirements.txt
```
