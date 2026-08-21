"""Runner script for Batch 4 synthetic unit tests."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from tests.test_batch4_pca import (
    test_population_matrix_assembly,
    test_mean_vector_and_centering,
    test_svd_eigenvalues_and_variance,
    test_retained_modes_k_selection_95pct,
    test_eigenvector_orthogonality,
    test_encoding_and_decoding,
    test_full_rank_exact_reconstruction,
    test_reconstruction_rmse_calculation,
    test_shape_model_save_and_load,
    test_population_statistics_thresholds,
    test_landmark_18d_statistics,
    test_side_branch_statistics,
)

TEST_FUNCTIONS = [
    ("1. Population Shape Matrix assembly (174 x 126) & Ordering", test_population_matrix_assembly),
    ("2. Mean shape vector mu and matrix centering X_centered", test_mean_vector_and_centering),
    ("3. SVD singular values, eigenvalues lambda_j = s_j^2 / (N-1), variance ratio", test_svd_eigenvalues_and_variance),
    ("4. Retained modes k selection for >= 95% cumulative variance threshold", test_retained_modes_k_selection_95pct),
    ("5. Retained eigenvector basis orthogonality P_k^T * P_k == I_k", test_eigenvector_orthogonality),
    ("6. PCA encoding b = P_k^T * (x - mu) and decoding x_hat = mu + P_k * b", test_encoding_and_decoding),
    ("7. Full-rank (126 modes) exact reconstruction (RMSE < 1e-10)", test_full_rank_exact_reconstruction),
    ("8. Reconstruction RMSE calculation sqrt(mean((x - x_hat)^2))", test_reconstruction_rmse_calculation),
    ("9. Statistical Shape Model JSON & NPZ save/load serialization", test_shape_model_save_and_load),
    ("10. Level 1-6 population statistics & P2.5 - P97.5 percentiles", test_population_statistics_thresholds),
    ("11. Level 2 Landmark 18-D vector statistics & SVD (§5.2)", test_landmark_18d_statistics),
    ("12. Level 5 Side branch statistics per parent vessel (§5.5)", test_side_branch_statistics),
]

def run_all_tests():
    print("=" * 80)
    print(" RUNNING BATCH 4 SYNTHETIC UNIT TESTS")
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
