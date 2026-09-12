"""Runner script for Batch 6 synthetic unit tests."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from tests.test_batch6_motion import (
    test_contraction_curve,
    test_ellipsoid_radial_contraction,
    test_ellipsoid_longitudinal_shortening,
    test_basetoapex_torsion_gradient,
    test_deformed_3d_surface_reconstruction,
    test_ostial_offset_motion_decay,
    test_lmca_bifurcation_snapping_all_10_phases,
    test_4d_segment_self_intersection_test,
)

TEST_FUNCTIONS = [
    ("1. Periodic contraction curve s(phi) profile (s(0)=0, s(0.35)=1, s(1)=0)", test_contraction_curve),
    ("2. Radial contraction semi-axes a(phi), b(phi) 15% shrink at peak systole", test_ellipsoid_radial_contraction),
    ("3. Longitudinal shortening semi-axis c(phi) 10% shrink at peak systole", test_ellipsoid_longitudinal_shortening),
    ("4. Base-to-apex torsion gradient (10 deg base rotation, 0 deg apex)", test_basetoapex_torsion_gradient),
    ("5. Deformed 3D surface reconstruction and local basis shift", test_deformed_3d_surface_reconstruction),
    ("6. Ostial off-surface linear offset motion decay with radial contraction", test_ostial_offset_motion_decay),
    ("7. LMCA bifurcation snapping continuity across all 10 cardiac phases", test_lmca_bifurcation_snapping_all_10_phases),
    ("8. 1.0 mm segment self-intersection test across all 10 cardiac phases", test_4d_segment_self_intersection_test),
]

def run_all_tests():
    print("=" * 80)
    print(" RUNNING BATCH 6 SYNTHETIC UNIT TESTS")
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
