"""Runner script for Batch 3 synthetic unit tests."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from tests.test_batch3_surface import (
    test_compute_ellipse_axis_extents_rotated_30deg,
    test_coronary_ellipse_fit,
    test_iv_ellipse_fit,
    test_b_consistency_diagnostic,
    test_ellipsoid_surface_equation,
    test_outward_unit_normal,
    test_circumferential_tangent_near_poles,
    test_longitudinal_tangent,
    test_projection_and_local_deviation_vector,
    test_fixed_42_point_representation_and_126d_shape,
    test_axis_alignment_theta_0_and_90deg,
)

TEST_FUNCTIONS = [
    ("1. Rotated Coronary Ellipse extent (theta=30deg)", test_compute_ellipse_axis_extents_rotated_30deg),
    ("2. Coronary Ellipse fit on RCA+LCX ring", test_coronary_ellipse_fit),
    ("3. IV Ellipse fit on LAD centerline", test_iv_ellipse_fit),
    ("4. b-consistency relative error diagnostic", test_b_consistency_diagnostic),
    ("5. Ellipsoid surface equation (x/a)^2+(y/b)^2+(z/c)^2 == 1", test_ellipsoid_surface_equation),
    ("6. Outward unit normal calculation", test_outward_unit_normal),
    ("7. Circumferential tangent t_u near poles (v=0, v=pi)", test_circumferential_tangent_near_poles),
    ("8. Longitudinal tangent t_v base-to-apex", test_longitudinal_tangent),
    ("9. Projection + 3D local deviation vector (dev_z preserved)", test_projection_and_local_deviation_vector),
    ("10. Fixed 42-point representation & 126-D shape vector", test_fixed_42_point_representation_and_126d_shape),
    ("11. Axis alignment theta=0deg and theta=90deg swapping", test_axis_alignment_theta_0_and_90deg),
]

def run_all_tests():
    print("=" * 80)
    print(" RUNNING BATCH 3 SYNTHETIC UNIT TESTS")
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
