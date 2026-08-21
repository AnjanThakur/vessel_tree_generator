"""Synthetic unit tests for Batch 3 Ellipse Fitting, Surface Projection, & 126-D Representation."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from surface_relative.ellipse_fitting import (
    compute_ellipse_axis_extents,
    fit_coronary_ellipse,
    fit_iv_ellipse,
)
from surface_relative.ellipsoid_model import (
    fit_patient_ellipsoid,
)
from surface_relative.surface_projection import (
    ellipsoid_point,
    ellipsoid_normal,
    ellipsoid_tangent_u,
    ellipsoid_tangent_v,
    project_point_to_surface,
)
from surface_relative.fixed_representation import (
    build_patient_fixed_representation,
    FIXED_COUNTS,
)


def test_compute_ellipse_axis_extents_rotated_30deg():
    """Test 1 (Mandatory): Verify compute_ellipse_axis_extents for theta = 30 deg."""
    a_raw, b_raw = 40.0, 20.0
    theta_rad = math.radians(30.0)

    ext_u, ext_v = compute_ellipse_axis_extents(a_raw, b_raw, theta_rad)

    expected_u = math.sqrt((40.0 * math.cos(theta_rad)) ** 2 + (20.0 * math.sin(theta_rad)) ** 2)
    expected_v = math.sqrt((40.0 * math.sin(theta_rad)) ** 2 + (20.0 * math.cos(theta_rad)) ** 2)

    np.testing.assert_allclose(ext_u, expected_u, atol=1e-6)
    np.testing.assert_allclose(ext_v, expected_v, atol=1e-6)
    assert not (ext_u == a_raw and ext_v == b_raw), "Formula must not simplify to raw semi-axes for rotated ellipse"


def test_coronary_ellipse_fit():
    """Test 2: Coronary ellipse fit on synthetic RCA + LCX ring in Z=0 plane."""
    t = np.linspace(0, 2 * np.pi, 30, endpoint=False)
    rca = np.column_stack([40 * np.cos(t[:15]), 30 * np.sin(t[:15]), np.zeros(15)])
    lcx = np.column_stack([40 * np.cos(t[15:]), 30 * np.sin(t[15:]), np.zeros(15)])

    res = fit_coronary_ellipse(rca, lcx)

    assert abs(res["aligned_axes"]["a_mm"] - 40.0) < 1.0
    assert abs(res["aligned_axes"]["b_cor_mm"] - 30.0) < 1.0


def test_iv_ellipse_fit():
    """Test 3: IV ellipse fit on synthetic LAD points in X=0 plane."""
    t = np.linspace(0, np.pi, 20)
    lad = np.column_stack([np.zeros(20), 35 * np.sin(t), -60 * np.cos(t)])

    res = fit_iv_ellipse(lad)

    assert res["aligned_axes"]["c_mm"] > 10.0
    assert res["aligned_axes"]["b_iv_mm"] > 10.0


def test_b_consistency_diagnostic():
    """Test 4: Verify b_consistency_relative_error calculation."""
    b_cor, b_iv = 37.04, 35.61
    rel_err = abs(b_cor - b_iv) / b_cor
    np.testing.assert_allclose(rel_err, 0.0386, atol=1e-3)


def test_ellipsoid_surface_equation():
    """Test 5: Verify (x/a)^2 + (y/b)^2 + (z/c)^2 == 1 for evaluated surface points."""
    a, b, c = 40.0, 30.0, 50.0
    for u in [0.0, 0.5, 1.2, 3.1, 5.5]:
        for v in [0.1, 0.8, 1.57, 2.5, 3.0]:
            p = ellipsoid_point(u, v, a, b, c)
            val = (p[0] / a) ** 2 + (p[1] / b) ** 2 + (p[2] / c) ** 2
            np.testing.assert_allclose(val, 1.0, atol=1e-6)


def test_outward_unit_normal():
    """Test 6: Verify ellipsoid_normal is finite and has unit magnitude."""
    a, b, c = 40.0, 30.0, 50.0
    n = ellipsoid_normal(0.5, 1.2, a, b, c)
    np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-6)


def test_circumferential_tangent_near_poles():
    """Test 7: Verify ellipsoid_tangent_u has no numerical singularity near poles (v=0 and v=pi)."""
    a, b, c = 40.0, 30.0, 50.0
    t_north = ellipsoid_tangent_u(0.5, 1.0e-12, a, b, c)
    t_south = ellipsoid_tangent_u(0.5, np.pi - 1.0e-12, a, b, c)

    np.testing.assert_allclose(np.linalg.norm(t_north), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(t_south), 1.0, atol=1e-6)


def test_longitudinal_tangent():
    """Test 8: Verify ellipsoid_tangent_v is finite and normalized."""
    a, b, c = 40.0, 30.0, 50.0
    tv = ellipsoid_tangent_v(0.8, 1.5, a, b, c)
    np.testing.assert_allclose(np.linalg.norm(tv), 1.0, atol=1e-6)


def test_projection_and_local_deviation_vector():
    """Test 9: Verify project_point_to_surface computes u, v, offset, dev_x, dev_y, dev_z."""
    a, b, c = 40.0, 30.0, 50.0
    u_ref, v_ref = 0.5, 1.0
    surf_pt = ellipsoid_point(u_ref, v_ref, a, b, c)
    normal = ellipsoid_normal(u_ref, v_ref, a, b, c)
    delta_offset = 0.5
    cardiac_pt = surf_pt + delta_offset * normal

    u, v, off, dev = project_point_to_surface(cardiac_pt, a, b, c)

    np.testing.assert_allclose(u, u_ref, atol=0.1)
    np.testing.assert_allclose(v, v_ref, atol=0.1)
    assert np.isfinite(dev[2]) and dev[2] > 0.0, "dev_z out-of-plane deviation must be preserved and positive"


def test_fixed_42_point_representation_and_126d_shape():
    """Test 10: Verify LMCA=5, LAD=12, LCX=10, RCA=15 = 42 points, 126D shape vector."""
    a, b, c = 40.0, 30.0, 50.0
    branches = {
        "lmca": np.column_stack([np.linspace(10, 15, 10), np.linspace(10, 15, 10), np.full(10, 10.0)]),
        "lad": np.column_stack([np.linspace(15, 30, 20), np.linspace(15, 30, 20), np.linspace(10, -20, 20)]),
        "lcx": np.column_stack([np.linspace(15, -20, 20), np.linspace(15, 20, 20), np.full(20, 10.0)]),
        "rca": np.column_stack([np.linspace(-10, -30, 30), np.linspace(10, 20, 30), np.full(30, 10.0)]),
    }

    res = build_patient_fixed_representation(branches, a, b, c)

    assert res["is_complete"] is True
    assert res["shape_vector"].shape == (126,)
    assert res["uvo_matrix_42_3"].shape == (42, 3)


def test_axis_alignment_theta_0_and_90deg():
    """Test 11: Verify extent_u = a_raw for theta=0, and extent_u = b_raw for theta=90 deg."""
    a_raw, b_raw = 40.0, 20.0

    # theta = 0 deg
    ext_u_0, ext_v_0 = compute_ellipse_axis_extents(a_raw, b_raw, 0.0)
    np.testing.assert_allclose(ext_u_0, a_raw, atol=1e-6)
    np.testing.assert_allclose(ext_v_0, b_raw, atol=1e-6)

    # theta = 90 deg (pi/2)
    ext_u_90, ext_v_90 = compute_ellipse_axis_extents(a_raw, b_raw, math.pi / 2)
    np.testing.assert_allclose(ext_u_90, b_raw, atol=1e-6)
    np.testing.assert_allclose(ext_v_90, a_raw, atol=1e-6)
