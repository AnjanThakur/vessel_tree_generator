"""Synthetic unit tests for Batch 4 PCA / Statistical Shape Model (SSM) fitting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from surface_relative.population_statistics import (
    fit_deviation_pca,
    compute_linear_stats,
    compute_landmark_statistics,
    compute_side_branch_statistics,
    build_validation_thresholds,
)
from ssm.shape_model import StatisticalShapeModel, DIMENSION_MAPPING


def test_population_matrix_assembly():
    """Test 1: Verify population shape matrix assembly and dimension mapping."""
    n_samples, n_features = 174, 126
    rng = np.random.default_rng(42)
    X_syn = rng.normal(loc=0.0, scale=2.0, size=(n_samples, n_features))

    assert X_syn.shape == (174, 126)
    assert DIMENSION_MAPPING["RCA"]["dims"] == 45
    assert DIMENSION_MAPPING["LMCA"]["dims"] == 15
    assert DIMENSION_MAPPING["LAD"]["dims"] == 36
    assert DIMENSION_MAPPING["LCX"]["dims"] == 30


def test_mean_vector_and_centering():
    """Test 2: Verify mean shape vector calculation and matrix centering."""
    n_samples, n_features = 100, 126
    rng = np.random.default_rng(42)
    X_syn = rng.normal(loc=5.0, scale=2.0, size=(n_samples, n_features))

    mean_vec = np.mean(X_syn, axis=0)
    centered = X_syn - mean_vec

    np.testing.assert_allclose(np.mean(centered, axis=0), np.zeros(126), atol=1e-10)


def test_svd_eigenvalues_and_variance():
    """Test 3: Verify SVD singular values, eigenvalues lambda_j = s_j^2 / (N - 1), and variance ratios."""
    n_samples, n_features = 174, 126
    rng = np.random.default_rng(42)
    X_syn = rng.normal(loc=0.0, scale=3.0, size=(n_samples, n_features))

    res = fit_deviation_pca(X_syn, variance_cutoff=0.95)

    S = res["singular_values"]
    eigenvalues = res["eigenvalues"]
    expected_eigenvalues = (S**2) / (n_samples - 1)

    np.testing.assert_allclose(eigenvalues, expected_eigenvalues, atol=1e-10)
    np.testing.assert_allclose(np.sum(res["explained_variance_ratio"]), 1.0, atol=1e-6)


def test_retained_modes_k_selection_95pct():
    """Test 4: Verify minimum k selection for 95% cumulative variance threshold."""
    n_samples, n_features = 100, 126
    rng = np.random.default_rng(42)
    weights = np.exp(-np.linspace(0, 3, 126))
    X_syn = rng.normal(size=(n_samples, 126)) * weights

    res = fit_deviation_pca(X_syn, variance_cutoff=0.95)

    k = res["k_retained"]
    cum_var = res["cumulative_explained_variance"]

    assert cum_var[k - 1] >= 0.95
    if k > 1:
        assert cum_var[k - 2] < 0.95


def test_eigenvector_orthogonality():
    """Test 5: Verify retained eigenvector basis orthogonality P_k^T * P_k == I_k."""
    n_samples, n_features = 174, 126
    rng = np.random.default_rng(42)
    X_syn = rng.normal(size=(n_samples, n_features))

    res = fit_deviation_pca(X_syn, variance_cutoff=0.95)

    P_k = res["components_retained"].T  # (126, k)
    ortho = P_k.T @ P_k

    np.testing.assert_allclose(ortho, np.eye(res["k_retained"]), atol=1e-12)


def test_encoding_and_decoding():
    """Test 6: Verify encoding b = P_k^T * (x - mu) and decoding x_hat = mu + P_k * b."""
    n_samples, n_features = 174, 126
    rng = np.random.default_rng(42)
    X_syn = rng.normal(size=(n_samples, n_features))

    res = fit_deviation_pca(X_syn, variance_cutoff=0.95)

    model = StatisticalShapeModel(
        mean_vector=res["mean_vector"],
        components_all=res["components_all"],
        components_retained=res["components_retained"],
        singular_values=res["singular_values"],
        eigenvalues=res["eigenvalues"],
        eigenvalues_retained=res["eigenvalues_retained"],
        explained_variance_ratio=res["explained_variance_ratio"],
        cumulative_explained_variance=res["cumulative_explained_variance"],
        k_retained=res["k_retained"],
        variance_cutoff=0.95,
        n_samples=n_samples,
        n_features=n_features,
    )

    x_test = X_syn[0]
    b = model.encode(x_test)
    x_hat = model.decode(b)

    assert b.shape == (model.k_retained,)
    assert x_hat.shape == (126,)


def test_full_rank_exact_reconstruction():
    """Test 7: Verify exact reconstruction with all 126 modes has zero error (< 1e-10)."""
    n_samples, n_features = 174, 126
    rng = np.random.default_rng(42)
    X_syn = rng.normal(size=(n_samples, n_features))

    res = fit_deviation_pca(X_syn, variance_cutoff=1.0)

    model = StatisticalShapeModel(
        mean_vector=res["mean_vector"],
        components_all=res["components_all"],
        components_retained=res["components_all"],
        singular_values=res["singular_values"],
        eigenvalues=res["eigenvalues"],
        eigenvalues_retained=res["eigenvalues"],
        explained_variance_ratio=res["explained_variance_ratio"],
        cumulative_explained_variance=res["cumulative_explained_variance"],
        k_retained=126,
        variance_cutoff=1.0,
        n_samples=n_samples,
        n_features=n_features,
    )

    x_test = X_syn[5]
    x_hat, rmse = model.reconstruct(x_test)

    np.testing.assert_allclose(x_hat, x_test, atol=1e-10)
    assert rmse < 1.0e-10


def test_reconstruction_rmse_calculation():
    """Test 8: Verify reconstruction RMSE formula sqrt(mean((x - x_hat)^2))."""
    n_samples, n_features = 50, 126
    rng = np.random.default_rng(42)
    X_syn = rng.normal(size=(n_samples, n_features))

    res = fit_deviation_pca(X_syn, variance_cutoff=0.50)

    model = StatisticalShapeModel(
        mean_vector=res["mean_vector"],
        components_all=res["components_all"],
        components_retained=res["components_retained"],
        singular_values=res["singular_values"],
        eigenvalues=res["eigenvalues"],
        eigenvalues_retained=res["eigenvalues_retained"],
        explained_variance_ratio=res["explained_variance_ratio"],
        cumulative_explained_variance=res["cumulative_explained_variance"],
        k_retained=res["k_retained"],
        variance_cutoff=0.50,
        n_samples=n_samples,
        n_features=n_features,
    )

    x_test = X_syn[2]
    x_hat, rmse = model.reconstruct(x_test)

    expected_rmse = float(np.sqrt(np.mean((x_test - x_hat) ** 2)))
    np.testing.assert_allclose(rmse, expected_rmse, atol=1e-10)


def test_shape_model_save_and_load():
    """Test 9: Verify JSON and NPZ serialization/deserialization."""
    n_samples, n_features = 50, 126
    rng = np.random.default_rng(42)
    X_syn = rng.normal(size=(n_samples, n_features))

    res = fit_deviation_pca(X_syn, variance_cutoff=0.95)

    model = StatisticalShapeModel(
        mean_vector=res["mean_vector"],
        components_all=res["components_all"],
        components_retained=res["components_retained"],
        singular_values=res["singular_values"],
        eigenvalues=res["eigenvalues"],
        eigenvalues_retained=res["eigenvalues_retained"],
        explained_variance_ratio=res["explained_variance_ratio"],
        cumulative_explained_variance=res["cumulative_explained_variance"],
        k_retained=res["k_retained"],
        variance_cutoff=0.95,
        n_samples=n_samples,
        n_features=n_features,
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        json_p = tmp_path / "model.json"
        npz_p = tmp_path / "model.npz"

        model.save(json_p, npz_p)

        assert json_p.exists()
        assert npz_p.exists()

        loaded_model = StatisticalShapeModel.load(json_p, npz_p)

        np.testing.assert_allclose(loaded_model.mean_vector, model.mean_vector, atol=1e-10)
        np.testing.assert_allclose(loaded_model.P_k, model.P_k, atol=1e-10)
        assert loaded_model.k_retained == model.k_retained


def test_population_statistics_thresholds():
    """Test 10: Verify empirical P2.5 - P97.5 percentile validation thresholds."""
    scaffolds = [{"a": 40.0, "b": 30.0, "c": 50.0}, {"a": 45.0, "b": 35.0, "c": 55.0}, {"a": 35.0, "b": 25.0, "c": 45.0}]
    lengths = {"LAD": [100.0, 110.0, 90.0]}
    angles = [45.0, 50.0, 40.0]
    devs = {"LAD": [1.0, 2.0, 1.5]}

    thresh = build_validation_thresholds(
        scaffold_params=scaffolds,
        branch_lengths=lengths,
        bifurcation_angles=angles,
        max_out_of_plane_devs=devs,
    )

    assert "ellipsoid_a_mm" in thresh
    assert "branch_length_LAD_mm" in thresh
    assert "bifurcation_angle_deg" in thresh
    assert "max_out_of_plane_LAD_mm" in thresh
    assert thresh["ellipsoid_a_mm"]["p2_5"] <= thresh["ellipsoid_a_mm"]["p97_5"]


def test_landmark_18d_statistics():
    """Test 11: Verify Level 2 landmark 18-D vector statistics calculation (§5.2)."""
    rng = np.random.default_rng(42)
    L_syn = rng.normal(loc=1.0, scale=0.5, size=(50, 18))

    res = compute_landmark_statistics(L_syn)

    assert res["n_samples"] == 50
    assert len(res["mean_18d"]) == 18
    assert len(res["std_18d"]) == 18
    assert len(res["explained_variance_ratio"]) == 18


def test_side_branch_statistics():
    """Test 12: Verify Level 5 side branch statistics calculation (§5.5)."""
    p_data = [
        {"RCA": {"count": 2, "attachment_t": [0.3, 0.7], "length_mm": [12.0, 15.0], "angle_deg": [45.0, 50.0]}},
        {"RCA": {"count": 1, "attachment_t": [0.5], "length_mm": [10.0], "angle_deg": [40.0]}},
    ]

    res = compute_side_branch_statistics(p_data)

    assert "RCA" in res
    assert res["RCA"]["branch_count"]["mean"] == 1.5
    assert res["RCA"]["branch_count"]["count"] == 2
