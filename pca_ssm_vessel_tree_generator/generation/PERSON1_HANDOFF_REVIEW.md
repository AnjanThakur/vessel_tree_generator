# Person 1 handoff review

Reviewed commit: `892b860`.

## Accepted interfaces

- `surface_relative.surface_projection` provides the shared ellipsoid point, normal, and tangent API required by Person 2.
- Cardiac-frame, ellipsoid, fixed-representation, population-statistics, and integrity modules are separated from generation code.
- The proposed fixed PCA contract is 126 dimensions ordered as RCA 15, LMCA 5, LAD 12, and LCX 10 points, with three local deviation components per point.

## Initial handoff gaps observed

- The committed `lca_ssm` entry is a gitlink, but the repository has no `.gitmodules` mapping. It therefore cannot be initialized as a submodule.
- `run_person1_week1_pipeline.py` reads `REPO_ROOT/lca_ssm/centerlines` and `REPO_ROOT/lca_ssm/nii files`; the pulled `lca_ssm` directory is empty in a normal checkout.
- A smoke run finds no centerline archives and then indexes the empty `cardiac_frames_rows` list while writing CSV headers, raising `IndexError`.
- None of the advertised generator-ready statistical output files were included in the handoff commit.
- The current population statistics are pooled surface-coordinate summaries; branch/landmark-specific frozen sampling distributions required by Person 2 are not yet present.

## Resolution in this increment

- The empty, unconfigured `lca_ssm` gitlink was removed. No source data were stored in it.
- `run_person1_week1_pipeline.py` now reads the 191 immutable archives under `outputs/lca_ssm/raw_cases` and verifies their archive and array hashes against the saved per-case records.
- The pipeline consumes the saved PPT planes and ellipse parameters without refitting or altering source centerlines.
- The pipeline reads the authoritative branch-assignment map before computing any branch-role statistics. Of 191 audited cases, 14 have resolved assignments and 6 also pass the raw anatomy and shape-preserving scaffold gates.
- Patient-specific cardiac frames, ellipsoid parameters, exact fixed cardiac controls, local surface deviations, landmark distributions, validation thresholds, and one joint 126-D deviation PCA are generated for those 6 eligible cases.
- The complete run retains 4 PCA modes and explains 96.743% cumulative variance. No unresolved assignment is used for statistics or PCA.
- Source coordinate and segment-length changes were both 0 mm, and all protected inputs were unchanged after the run.
- A frozen package is written to `outputs/lca_ssm/person1_week1/generator_statistics`, including a SHA-256 manifest for every required artifact.
- The generator package includes exact case-matched cardiac controls and deviation arrays. Person 2 reconstructs each baseline directly in cardiac XYZ and uses PCA only for low-scale coordinated innovation.
- Real-derived progression limits record backward progress, terminal progress, and maximum local turn for loop detection.

Person 2 real-statistics mode now consumes that frozen package. Ellipsoid axes, landmarks, and fixed trajectories are selected from the same patient identifier to preserve their empirical dependence; centered PCA variation then supplies coordinated residual shape variation without pointwise noise.

## Remaining scientific limitation

The frozen package is a generator handoff, not proof that generated and real distributions are identical. The completed primary Person 2 cohort contains 25 individually accepted LMCA/LAD/LCX trees and compares 41 scaffold and PCA measures against the 6 resolved, anatomy-gated references. All 191 source cases remain visible in the audit, but unresolved assignments are not treated as branch-labelled truth. Descriptive warnings remain because the reference cohort is small and smooth scaffolds differ from noisy full-resolution centerlines; warnings are retained in the final validation JSON and CSV. RCA is excluded from the primary cohort because the available paths are inferred disconnected candidates rather than resolved RCA ground truth.
