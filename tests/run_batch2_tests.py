"""Runner script for Batch 2 synthetic unit tests."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from tests.test_batch2_alignment import (
    test_coronary_plane_svd_planar_ring,
    test_normal_sign_orientation_lad_descent,
    test_iv_normal_cross_product,
    test_right_handed_cardiac_frame,
    test_scanner_to_cardiac_rigid_transform,
    test_length_preservation_invariant,
    test_lca_canonical_z_rotation,
    test_degenerate_collinear_plane_input,
    test_zero_lad_direction_input,
    test_rca_unresolved_case_handling,
)

TEST_FUNCTIONS = [
    ("1. Coronary plane SVD planar fit", test_coronary_plane_svd_planar_ring),
    ("2. Normal sign orientation (LAD descent)", test_normal_sign_orientation_lad_descent),
    ("3. IV normal cross-product construction", test_iv_normal_cross_product),
    ("4. Right-handed cardiac frame (det=+1)", test_right_handed_cardiac_frame),
    ("5. Scanner -> cardiac rigid transform", test_scanner_to_cardiac_rigid_transform),
    ("6. Length preservation invariant (<1e-5mm)", test_length_preservation_invariant),
    ("7. LCA canonical Z-rotation (angle=0, +X)", test_lca_canonical_z_rotation),
    ("8. Degenerate collinear plane error handling", test_degenerate_collinear_plane_input),
    ("9. Zero LAD direction error handling", test_zero_lad_direction_input),
    ("10. RCA-unresolved case handling", test_rca_unresolved_case_handling),
]

def run_all_tests():
    print("=" * 80)
    print(" RUNNING BATCH 2 SYNTHETIC UNIT TESTS")
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
