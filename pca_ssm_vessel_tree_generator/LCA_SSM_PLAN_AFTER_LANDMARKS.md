# LCA SSM plan after landmark extraction

## Scope

This stage converts the imported PCA label cohort into fitted LCA geometry parameters for a later 27-control-point generator. It reads the existing NIfTI labels and centerline graphs and writes only derived files under `outputs/lca_ssm/`. It does not change disease, radius tapering, tube surfaces, tight meshes, hub connectors, or the existing anatomical pipeline.

Run from the repository root:

```bash
python pca_ssm_vessel_tree_generator/build_lca_ssm_population_stats.py --clean
```

The default inputs are:

- `pca_ssm_vessel_tree_generator/nii files/<case>.nii.gz`
- `pca_ssm_vessel_tree_generator/centerlines/<case>/summary.csv`
- `pca_ssm_vessel_tree_generator/centerlines/<case>/branches.npz`

There are 200 matched cases. `--input-dir`, `--nifti-dir`, `--output-dir`, `--workers`, and `--max-patients` support controlled runs and smoke tests.

## Source-schema findings

Every centerline folder contains a skan branch table and the corresponding ordered path arrays. Only one folder contains the older experimental `landmarks.npy` and pruned files, so those optional files cannot be the population contract. The pipeline instead derives endpoints and junctions consistently from every `summary.csv` graph.

All 200 NIfTI files have a valid sform, axis-aligned RAS orientation, positive label content, and matching centerline coordinates. Shapes vary in the cropped z dimension, but all have 512 by 512 in-plane dimensions. The adapter streams the compressed float64 NIfTI data one z-plane at a time and binarizes positive values into a Boolean mask to avoid holding a full float volume in memory.

## LCA extraction from each label

The adapter in `lca_ssm_label_adapter.py` deliberately preserves the selection logic already established by stages 1 and 2:

1. Build the graph from `summary.csv`, retaining the shorter path if duplicate graph edges exist.
2. Find graph endpoints and sample their physical vessel radius from the label mask with a local 3-D Euclidean distance transform.
3. Select the endpoint with the largest radius as the candidate root. Ties are deterministic by node ID and are reported through a zero root-radius margin.
4. Build the root-directed shortest-path tree.
5. Evaluate junctions using the existing thresholds: second daughter at least 10 mm, daughter-length ratio at least 0.25, and root-to-bifurcation distance at least 5 mm.
6. Rank accepted junctions by the existing rule: largest second daughter, then shorter root distance.
7. Extract ordered LMCA and the two dominant daughter paths from `branches.npz`.
8. Transform their points into world millimetres through the case's NIfTI sform.

Cases are rejected rather than relaxed or repaired when no bifurcation passes these rules.

## LAD and LCX identity

The 200-case source does not contain manual LAD/LCX daughter annotations. Branch A is merely the longer selected daughter and cannot always be renamed LAD. To use this cohort as requested without pretending labels exist, the adapter records the daughters as anatomically inferred candidates.

The common RAS frame provides two documented reference directions:

- LAD reference: anterior-inferior, normalized from `[0, 1, -1]`.
- LCX reference: patient-left, `[-1, 0, 0]`.

Both possible assignments are scored from 75% overall branch direction and 25% initial tangent alignment. A small 0.10 prior favors the longer daughter as LAD. The assignment with the greater combined LAD-plus-LCX score is selected. Every per-patient JSON stores both candidate features, both assignment scores, the chosen sources, and their score margin. A margin below 0.20 is flagged for review but is not silently discarded.

This is reproducible anatomical inference, not ground-truth labelling. The current cohort assigns Branch A as LAD in 124 accepted cases and Branch B as LAD in 67, demonstrating that classification is not a length-only rename. Manual or metadata-based validation remains important, especially for low-margin cases.

## Plane and curve fitting

For each accepted tree:

1. Fit the LCX coronary/AV-groove plane by least-squares SVD. Save its centroid, deterministic in-plane basis, normal, singular values, and every absolute point-to-plane residual.
2. Project LCX into that plane and fit an ordered robust nonlinear ellipse arc. Save center, semi-axes, axis ratio, orientation, unwrapped angular positions, start/end angle, extent, numerical arc length, and per-point residuals.
3. If the ellipse optimizer is unavailable or yields an unstable, overly eccentric, or high-residual solution, use the recorded PCA/quadratic arc fallback.
4. Fit the LAD interventricular plane by SVD and project LAD into it.
5. Fit an ordered quadratic LAD guide against normalized cumulative arc position. Save polynomial coefficients, guide length, and per-point residuals.

The pipeline also derives LMCA/LAD/LCX lengths, chords, tortuosities, initial and distal tangents, total turning, mean curvature, branch-length ratios, bifurcation angle, LMCA continuity angles, plane angle, and RAS-based descent/circumflex scores.

## Validation and current results

Checks include:

- matching NIfTI, summary, and branch archive exist;
- NIfTI header, sform, mask, coordinates, root, and graph are valid;
- the established bifurcation thresholds are satisfied;
- LMCA has at least two points and each daughter has at least five;
- all coordinates and derived values are finite;
- branches have non-zero length and chord;
- branch endpoints agree at the selected junction;
- LAD and LCX are non-collinear enough for stable planes;
- angles and curve fits are finite;
- all point-to-plane and point-to-curve residuals are retained.

The full audit accepts 191 cases and rejects 9: `65.label`, `75.label`, `80.label`, `97.label`, `130.label`, `149.label`, `165.label`, `171.label`, and `176.label`. Each rejected case has the same evidence-backed reason: no junction satisfies the existing stage-2 LCA selection thresholds.

Accepted cases are not all equally certain. The population statistics report assignment-margin and root-radius-margin distributions, low-margin patient IDs, and whether Branch A or Branch B supplied the inferred LAD. Review warnings do not alter the source geometry.

## Output contract

```text
outputs/lca_ssm/
|-- valid_patient_index.json
|-- rejected_patient_index.json
|-- run_manifest.json
|-- ssm_population_statistics.json
|-- ssm_population_statistics.csv
|-- ssm_generator_schema.json
|-- patients/
|   |-- <case>/ssm_parameters.json
|   |-- <rejected-case>/rejection.json
|-- visualizations/
|   |-- <case>_planes_3d.png
|   |-- <case>_lcx_fit.png
|   |-- <case>_lad_fit.png
|   |-- population_parameter_summary.png
```

Per-patient files retain source-selection evidence, ostium/bifurcation/terminals, fixed generator landmark positions, detailed plane and curve fits, per-point deviations, and quality warnings. The population files contain count, mean, sample standard deviation, extrema, and 5/25/50/75/95 percentiles for scalar, vector, control-position, and pooled noise parameters.

The generator schema preserves branch-local targets of LMCA 5, LAD 12, and LCX 10 (27 points total). It exposes observed and recommended ranges but warns a future generator to preserve joint covariance and renormalize sampled direction vectors.

## Next step

The next generator stage should align and resample each accepted tree to the fixed topology, fit a joint covariance-aware PCA model, validate held-out reconstruction, and generate 27-point LCA control trees. Low-margin daughter assignments should be manually or metadata-validated before treating LAD/LCX identity as ground truth. Circular ellipse angles also require circular rather than ordinary linear statistics in a production sampler.
