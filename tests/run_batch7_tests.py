"""Runner script for Batch 7 synthetic unit tests."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from tests.test_batch7_output import (
    test_resample_curve_3d,
    test_compute_vessel_radius,
    test_deterministic_branch_ordering,
    test_construct_geometry_static_array,
    test_construct_geometry_cine_array,
    test_static_equals_cine0_invariant,
    test_build_global_pipeline_report,
    test_pipeline_cli_integration,
)

TEST_FUNCTIONS = [
    ("1. 3D curve arc-length resampling to N_points=50 with endpoint preservation", test_resample_curve_3d),
    ("2. Anatomical vessel radius model assignment (LMCA=2.0, RCA/LAD/LCX=1.8->0.8, SB=0.6->0.4)", test_compute_vessel_radius),
    ("3. Deterministic branch ordering (0=RCA, 1=LMCA, 2=LAD, 3=LCX, 4+=side branches)", test_deterministic_branch_ordering),
    ("4. Static geometry array construction (M_branches x 50 x 4) and radius r > 0", test_construct_geometry_static_array),
    ("5. 4D cine geometry array construction (10 x M_branches x 50 x 4)", test_construct_geometry_cine_array),
    ("6. Static vs. Cine phi=0.0 frame exact identity invariant (atol=1e-12)", test_static_equals_cine0_invariant),
    ("7. Dynamic end-to-end pipeline report aggregation from output artifacts", test_build_global_pipeline_report),
    ("8. Non-invasive master pipeline CLI parser integration", test_pipeline_cli_integration),
]

def run_all_tests():
    print("=" * 80)
    print(" RUNNING BATCH 7 SYNTHETIC UNIT TESTS")
    print("=" * 80)
    passed = 0
    failed = 0
    for name, test_fn in TEST_FUNCTIONS:
        try:
            test_fn()
            print(f" [PASS] {name}")
            passed += 1
        except Exception as exc:
            print(f" [FAIL] {name}: {exc}")
            failed += 1
    print("=" * 80)
    print(f" RESULTS: {passed} PASSED, {failed} FAILED out of {len(TEST_FUNCTIONS)} tests")
    print("=" * 80)
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    raise SystemExit(run_all_tests())
