"""Runner script for Batch 5 synthetic unit tests."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from tests.test_batch5_generation import (
    test_ellipsoid_scaffold_sampling,
    test_landmark_sampling,
    test_bspline_surface_fitting,
    test_obliquity_and_tortuosity_perturbation,
    test_ssm_34mode_deviations_sampling,
    test_3d_surface_coordinate_evaluation,
    test_ostial_linear_offset_decay,
    test_lmca_bifurcation_snapping,
    test_side_branch_random_surface_walk,
    test_3d_segment_self_intersection_detector,
)

TEST_FUNCTIONS = [
    ("1. Ellipsoid scaffold semi-axes sampling (a,b,c) truncated at +-2sigma", test_ellipsoid_scaffold_sampling),
    ("2. Landmark (u, v, offset) sampling within parameter domain bounds", test_landmark_sampling),
    ("3. 2D cubic B-spline curve fitting u(t), v(t) in parameter space", test_bspline_surface_fitting),
    ("4. Obliquity linear drift and zero-endpoint tortuosity envelope E(t)", test_obliquity_and_tortuosity_perturbation),
    ("5. 34-mode SSM deviation sampling & 126-D shape vector synthesis", test_ssm_34mode_deviations_sampling),
    ("6. 3D surface coordinate evaluation p = s(u,v) + dev_x*t_u + dev_y*t_v + dev_z*n", test_3d_surface_coordinate_evaluation),
    ("7. Ostial off-surface linear offset decay offset(t) = offset_ostium*(1-t)", test_ostial_linear_offset_decay),
    ("8. LMCA bifurcation snapping LAD[0] == LCX[0] == LMCA[-1]", test_lmca_bifurcation_snapping),
    ("9. Side branch Poisson count and attachment position sampling", test_side_branch_random_surface_walk),
    ("10. 1.0 mm 3D segment-to-segment self-intersection test & rejection flag", test_3d_segment_self_intersection_detector),
]

def run_all_tests():
    print("=" * 80)
    print(" RUNNING BATCH 5 SYNTHETIC UNIT TESTS")
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
