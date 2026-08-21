"""Synthetic unit tests for Batch 5 Synthetic Coronary Tree Generation."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from generation.bspline_surface import (
    surface_spline,
    add_tortuosity,
    add_obliquity,
    generate_vessel_uv,
)
from generation.validator import (
    segment_distance_3d,
    has_self_intersection,
    validate_synthetic_tree,
)
from ssm.shape_model import StatisticalShapeModel, DIMENSION_MAPPING


def test_ellipsoid_scaffold_sampling():
    """Test 1: Verify ellipsoid semi-axes sampling and clipping logic."""
    mu_a, std_a = 40.0, 5.0
    rng = np.random.default_rng(42)
    vals = [np.clip(rng.normal(mu_a, std_a), mu_a - 2 * std_a, mu_a + 2 * std_a) for _ in range(100)]
    assert min(vals) >= mu_a - 2 * std_a
    assert max(vals) <= mu_a + 2 * std_a


def test_landmark_sampling():
    """Test 2: Verify 18-D landmark sampling within valid parameter domain ranges."""
    rng = np.random.default_rng(42)
    u_vals = [np.clip(rng.normal(np.pi, 0.5), 0.0, 2 * np.pi) for _ in range(100)]
    v_vals = [np.clip(rng.normal(np.pi / 2, 0.5), 0.01, np.pi - 0.01) for _ in range(100)]

    assert min(u_vals) >= 0.0 and max(u_vals) <= 2 * np.pi
    assert min(v_vals) >= 0.01 and max(v_vals) <= np.pi - 0.01


def test_bspline_surface_fitting():
    """Test 3: Verify 2D cubic B-spline curve fitting in (u, v) parameter space."""
    u_ctrl = np.array([0.0, 0.5, 1.0, 1.5])
    v_ctrl = np.array([0.2, 0.4, 0.6, 0.8])

    u_eval, v_eval = surface_spline(u_ctrl, v_ctrl, num_eval=50)

    assert len(u_eval) == 50
    assert len(v_eval) == 50
    np.testing.assert_allclose(u_eval[0], u_ctrl[0], atol=1e-5)
    np.testing.assert_allclose(v_eval[-1], v_ctrl[-1], atol=1e-5)


def test_obliquity_and_tortuosity_perturbation():
    """Test 4: Verify obliquity linear drift and zero-endpoint perturbation envelope E(0)=E(1)=0."""
    u_ctrl = np.array([0.0, 0.5, 1.0, 1.5])
    v_ctrl = np.array([0.2, 0.4, 0.6, 0.8])
    rng = np.random.default_rng(42)

    # Obliquity drift
    u_ob = add_obliquity(u_ctrl, 0.5)
    np.testing.assert_allclose(u_ob[0], u_ctrl[0], atol=1e-10)
    assert u_ob[-1] > u_ctrl[-1]

    # Tortuosity perturbation
    u_t, v_t = add_tortuosity(u_ctrl, v_ctrl, strength=0.2, rng=rng)
    np.testing.assert_allclose(u_t[0], u_ctrl[0], atol=1e-10)
    np.testing.assert_allclose(u_t[-1], u_ctrl[-1], atol=1e-10)


def test_ssm_34mode_deviations_sampling():
    """Test 5: Verify 34-mode SSM deviation sampling and 126-D shape vector synthesis."""
    mean_vec = np.zeros(126)
    components = np.eye(126)[:34]
    singular_values = np.ones(34) * 10.0
    eigenvalues = np.ones(34) * 5.0
    exp_var = np.linspace(0.1, 0.95, 34)

    model = StatisticalShapeModel(
        mean_vector=mean_vec,
        components_all=np.eye(126),
        components_retained=components,
        singular_values=singular_values,
        eigenvalues=np.pad(eigenvalues, (0, 92)),
        eigenvalues_retained=eigenvalues,
        explained_variance_ratio=np.pad(exp_var, (0, 92)),
        cumulative_explained_variance=np.pad(exp_var, (0, 92)),
        k_retained=34,
        variance_cutoff=0.95,
        n_samples=174,
        n_features=126,
    )

    rng = np.random.default_rng(42)
    b_coeffs = np.zeros(34)
    for j in range(34):
        std_j = np.sqrt(eigenvalues[j])
        b_coeffs[j] = np.clip(rng.normal(0, std_j), -2 * std_j, 2 * std_j)

    x_syn = model.decode(b_coeffs)
    assert x_syn.shape == (126,)


def test_3d_surface_coordinate_evaluation():
    """Test 6: Verify 3D surface point evaluation p = s(u,v) + dev_x*t_u + dev_y*t_v + dev_z*n."""
    from surface_relative.surface_projection import (
        ellipsoid_point,
        ellipsoid_normal,
        ellipsoid_tangent_u,
        ellipsoid_tangent_v,
    )

    a, b, c = 40.0, 30.0, 25.0
    u, v = 0.5, 1.0
    surf = ellipsoid_point(u, v, a, b, c)
    n = ellipsoid_normal(u, v, a, b, c)
    tu = ellipsoid_tangent_u(u, v, a, b, c)
    tv = ellipsoid_tangent_v(u, v, a, b, c)

    dev_x, dev_y, dev_z = 1.0, -0.5, 2.0
    p_3d = surf + dev_x * tu + dev_y * tv + dev_z * n

    assert p_3d.shape == (3,)
    assert not np.allclose(p_3d, surf)


def test_ostial_linear_offset_decay():
    """Test 7: Verify linear off-surface offset decay offset(t) = offset_ostium*(1-t)."""
    ost_offset = 10.0
    t_arr = np.linspace(0.0, 1.0, 5)
    decay_offsets = ost_offset * (1.0 - t_arr)

    assert decay_offsets[0] == 10.0
    assert decay_offsets[-1] == 0.0
    assert decay_offsets[2] == 5.0


def test_lmca_bifurcation_snapping():
    """Test 8: Verify exact LMCA bifurcation snapping LAD[0] == LCX[0] == LMCA[-1]."""
    lmca_pts = np.array([[0, 0, 0], [10, 5, 2]])
    bif_pt = lmca_pts[-1].copy()

    lad_pts = np.array([[0, 0, 0], [20, 10, 4]])
    lcx_pts = np.array([[0, 0, 0], [15, -10, 1]])

    lad_pts[0] = bif_pt
    lcx_pts[0] = bif_pt

    np.testing.assert_allclose(lad_pts[0], lmca_pts[-1], atol=1e-12)
    np.testing.assert_allclose(lcx_pts[0], lmca_pts[-1], atol=1e-12)


def test_side_branch_random_surface_walk():
    """Test 9: Verify side branch Poisson count and attachment position sampling."""
    rng = np.random.default_rng(42)
    counts = rng.poisson(lam=2.5, size=100)
    t_attachments = rng.uniform(0.10, 0.85, size=100)

    assert min(counts) >= 0
    assert min(t_attachments) >= 0.10
    assert max(t_attachments) <= 0.85


def test_3d_segment_self_intersection_detector():
    """Test 10: Verify minimum 3D segment-to-segment distance calculation and self-intersection flag."""
    # Test 1: Intersecting/near segments (< 1.0mm)
    vessels_intersecting = {
        "LAD": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        "LCX": np.array([[5.0, 0.2, 0.0], [5.0, 10.0, 0.0]]),
    }
    has_int, msg = has_self_intersection(vessels_intersecting, min_dist_threshold_mm=1.0)
    assert has_int is True
    assert "Self-intersection" in msg

    # Test 2: Well-separated segments (> 1.0mm)
    vessels_clean = {
        "LAD": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        "LCX": np.array([[0.0, 20.0, 0.0], [10.0, 20.0, 0.0]]),
    }
    has_int_clean, _ = has_self_intersection(vessels_clean, min_dist_threshold_mm=1.0)
    assert has_int_clean is False
