# Coronary4D Independent Project Verification Guide

Run all commands from:

```text
C:\Internship\vessel_tree_generator
```

The presentation verifier never repairs failed evidence. Frozen population
statistics and protected source/PPT-derived geometry are read-only inputs.

## 1. Fast release verification

Start the presentation center and click **Run Release Verification**:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator present
```

Programmatic equivalent:

```powershell
.\.venv\Scripts\python.exe -c "from submission_release.presentation_center.verification import run_fast_verification; import json; print(json.dumps(run_fast_verification(), indent=2))"
```

Expected overall status: `PASS`.

## 2. Source integrity

Evidence:

```text
submission_release/final_audit/pre_fix_hash_manifest.json
submission_release/final_audit/protected_integrity_comparison.json
submission_release/presentation_center/verification_records/
```

Expected: 485/485 immutable evidence/model files unchanged; maximum protected
source coordinate and source segment-length changes both 0.0 mm.

## 3. Cohort counts and exclusions

Run:

```powershell
.\.venv\Scripts\python.exe pipeline.py audit-final
```

Primary evidence:

```text
submission_release/final_audit/cohort_funnel.csv
submission_release/final_audit/cohort_exclusion_reason_summary.csv
submission_release/final_audit/assignment_audit.csv
```

Expected funnel: `200 -> 191 -> 181 -> 65 -> 52`.

## 4. PCA/statistical model

Inspect:

```text
submission_release/final_audit/pca_recomputation_audit.json
outputs/lca_ssm/lca_population_model/generator_statistics/
```

Expected: matrix `[52, 81]`, 13 retained modes, cumulative variance about
0.955469, and independent recomputation PASS.

## 5. Generated population

Inspect:

```text
outputs/lca_ssm/lca_population_cohort/cohort_manifest.json
outputs/lca_ssm/lca_population_cohort/cohort_metrics.csv
```

Expected: 52 accepted trees in 64 attempts, all 52 eligible baselines
represented.

## 6. Exact topology

Fast verification loads each `geometry_cine.npy` and checks every phase:

```text
LMCA[-1, XYZ] == LAD[0, XYZ]
LMCA[-1, XYZ] == LCX[0, XYZ]
```

The accepted numerical tolerance is `1e-9 mm`; release maximum is effectively
zero. Topology metadata is also stored in each `graph.json`.

## 7. Disease identity

Inspect:

```text
submission_release/final_validation/disease_validation.json
submission_release/demo_cases/*/quantitative_validation.json
```

Expected: disease changes radius only and healthy-versus-disease XYZ difference
is 0.0 mm.

## 8. Motion and phase closure

Inspect:

```text
submission_release/final_validation/motion_validation.json
outputs/lca_ssm/lca_population_motion/4d_trees_summary.json
```

Expected: 52 trees, 10 stored frames, 520 snapshots, nine independent positions
plus repeated closure, topology preserved.

## 9. Pulsatility and lesion compliance

Inspect:

```text
submission_release/final_validation/disease_validation.json
submission_release/demo_cases/QUANTITATIVE_AUDIT_SUMMARY.json
```

Expected healthy global change: about 2.9916%. Lesion response is lower and is
reported separately for focal LAD, diffuse LCX, and tandem LAD.

## 10. Novelty

Run:

```powershell
.\.venv\Scripts\python.exe pipeline.py novelty-validate
```

Evidence:

```text
submission_release/final_audit/generation_novelty_summary.json
submission_release/final_audit/generation_novelty_audit.csv
```

Expected: zero exact duplicates and zero near duplicates below 0.1 mm.

## 11. Internal holdout

Run:

```powershell
.\.venv\Scripts\python.exe pipeline.py holdout-validate
```

Evidence:

```text
submission_release/final_audit/holdout_validation_summary.json
submission_release/final_audit/holdout_leakage_audit.csv
```

Expected: 41/52 accepted, 78.85% anatomical generation acceptance, zero source
leakage and zero baseline leakage. This is not classification accuracy or
external validation.

## 12. VTK

Run:

```powershell
.\.venv\Scripts\python.exe -m vessel_tree_generator verify --input-dir submission_release\demo_cases\focal_lad
```

Expected: PASS, including checksums, tensor shape, topology, and VTK readback.

## 13. Regression tests

Read-only commands:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m unittest pca_ssm_vessel_tree_generator.tests.test_person2_generation -q
.\.venv\Scripts\python.exe -m compileall -q vessel_tree_generator pca_ssm_vessel_tree_generator lca_vessel_tree_generator rca_vessel_tree_generator tests tools
.\.venv\Scripts\python.exe -m pip check
```

Final recorded release result: 90 passed, zero failed, compileall PASS, pip check
PASS.

## 14. Deep presentation verification

Click **Run Full Verification** or run:

```powershell
.\.venv\Scripts\python.exe -c "from submission_release.presentation_center.verification import run_full_verification; import json; print(json.dumps(run_full_verification(print), indent=2))"
```

This checks source integrity, PCA, generation, disease, motion, VTK,
novelty/holdout, and regression tests, then compares frozen hashes before and
after. It does not invoke any audit writer.
