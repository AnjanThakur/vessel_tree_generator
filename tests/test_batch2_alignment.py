"""Unit tests for Batch 2 Plane Fitting & Cardiac Frame Alignment."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

# Ensure project directory is in python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from pca_ssm_vessel_tree_generator.alignment.plane_fitting import (
    fit_coronary_plane,
    compute_iv_normal,
)
from pca_ssm_vessel_tree_generator.alignment.cardiac_frame_pipeline import (
    process_patient_cardiac_frame,
)
from surface_relative.cardiac_frame import (
    compute_cardiac_frame,
    transform_to_cardiac_frame,
    validate_cardiac_frame,
)


def test_coronary_plane_svd_planar_ring():
    """Test 1: SVD coronary plane fit on a synthetic planar ring in Z=50 plane."""
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    rca = np.column_stack([30 * np.cos(theta[:10]) + 100, 30 * np.sin(theta[:10]) + 100, np.full(10, 50.0)])
    lcx = np.column_stack([30 * np.cos(theta[10:]) + 100, 30 * np.sin(theta[10:]) + 100, np.full(10, 50.0)])
    lad = np.array([[100, 100, 50], [100, 100, 30], [100, 100, 10]])  # LAD descends in -Z

    res = fit_coronary_plane(rca, lcx, lad)

    np.testing.assert_allclose(res["coronary_centroid"], [100.0, 100.0, 50.0], atol=1e-5)
    assert abs(res["rms_residual_mm"]) < 1e-6
    assert abs(res["coronary_normal"][2]) > 0.99
    assert res["dot_after_correction"] < 0.0


def test_normal_sign_orientation_lad_descent():
    """Test 2: Verify SVD normal is reoriented so LAD descent vector . coronary_normal < 0."""
    theta = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    rca = np.column_stack([30 * np.cos(theta[:10]), 30 * np.sin(theta[:10]), np.zeros(10)])
    lcx = np.column_stack([30 * np.cos(theta[10:]), 30 * np.sin(theta[10:]), np.zeros(10)])
    
    # LAD descends from z=10 to z=-10 (descent vector = (0, 0, -20))
    lad = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 0.0], [0.0, 0.0, -10.0]])

    res = fit_coronary_plane(rca, lcx, lad)
    # coronary_normal must point in +Z so that lad_dir . coronary_normal = (0,0,-20) . (0,0,1) = -20 < 0
    assert res["coronary_normal"][2] > 0.0
    assert res["dot_after_correction"] < 0.0


def test_iv_normal_cross_product():
    """Test 3: Verify IV normal calculation via cross product unit(n_cor x v_LAD)."""
    cor_normal = np.array([0.0, 0.0, 1.0])
    lad_pts = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0]])  # LAD descends in Y and -Z

    res = compute_iv_normal(cor_normal, lad_pts)

    assert abs(res["orthogonality_error"]) < 1e-6
    np.testing.assert_allclose(np.linalg.norm(res["iv_normal"]), 1.0, atol=1e-6)


def test_right_handed_cardiac_frame():
    """Test 4: Verify det(R_total) = +1 and R_total @ R_total.T = I."""
    cor_normal = np.array([0.0, 0.0, 1.0])
    iv_normal = np.array([1.0, 0.0, 0.0])
    centroid = np.array([10.0, 20.0, 30.0])
    lad_pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, -10.0]])
    lca_ost = np.array([15.0, 20.0, 35.0])

    orig, R_total, meta = compute_cardiac_frame(cor_normal, iv_normal, centroid, lad_pts, lca_ost)

    det_val = float(np.linalg.det(R_total))
    np.testing.assert_allclose(det_val, 1.0, atol=1e-5)
    np.testing.assert_allclose(R_total @ R_total.T, np.eye(3), atol=1e-6)


def test_scanner_to_cardiac_rigid_transform():
    """Test 5: Rigid point cloud transformation (scanner -> cardiac)."""
    origin = np.array([10.0, 20.0, 30.0])
    R_total = np.eye(3)

    pts_scanner = np.array([[10.0, 20.0, 30.0], [20.0, 30.0, 40.0]])
    pts_cardiac = transform_to_cardiac_frame(pts_scanner, origin, R_total)

    np.testing.assert_allclose(pts_cardiac[0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(pts_cardiac[1], [10.0, 10.0, 10.0], atol=1e-6)


def test_length_preservation_invariant():
    """Test 6: Rigid transformation preserves Euclidean segment lengths strictly (< 1e-5 mm)."""
    cor_normal = np.array([0.0, 0.0, 1.0])
    iv_normal = np.array([1.0, 0.0, 0.0])
    centroid = np.array([5.0, 5.0, 5.0])
    lad_pts = np.array([[0.0, 0.0, 10.0], [0.0, 5.0, 0.0], [0.0, 10.0, -10.0]])
    lca_ost = np.array([10.0, 5.0, 10.0])

    orig, R_total, _ = compute_cardiac_frame(cor_normal, iv_normal, centroid, lad_pts, lca_ost)

    raw_pts = np.array([[1.0, 2.0, 3.0], [4.0, 8.0, 12.0], [9.0, 18.0, 27.0]])
    card_pts = transform_to_cardiac_frame(raw_pts, orig, R_total)

    raw_lens = np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)
    card_lens = np.linalg.norm(np.diff(card_pts, axis=0), axis=1)

    np.testing.assert_allclose(raw_lens, card_lens, atol=1e-6)


def test_lca_canonical_z_rotation():
    """Test 7: Verify canonical Z-rotation rotates LCA ostium to angle = 0 (+X axis)."""
    cor_normal = np.array([0.0, 0.0, 1.0])
    iv_normal = np.array([1.0, 0.0, 0.0])
    centroid = np.array([0.0, 0.0, 0.0])
    lad_pts = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]])
    # LCA ostium sits at (0, 15, 5) initially (angle = +90 deg / +Y axis)
    lca_ost = np.array([0.0, 15.0, 5.0])

    orig, R_total, meta = compute_cardiac_frame(cor_normal, iv_normal, centroid, lad_pts, lca_ost)
    lca_cardiac = transform_to_cardiac_frame(lca_ost, orig, R_total)

    # In canonical cardiac frame, LCA ostium Y component should be ~0 and X component > 0
    assert abs(lca_cardiac[1]) < 1e-5
    assert lca_cardiac[0] > 0.0


def test_degenerate_collinear_plane_input():
    """Test 8: Collinear ring points raise ValueError."""
    line_pts = np.column_stack([np.linspace(0, 10, 10), np.zeros(10), np.zeros(10)])
    lad_pts = np.array([[0, 0, 10], [0, 0, -10]])

    raised = False
    try:
        fit_coronary_plane(line_pts[:5], line_pts[5:], lad_pts)
    except ValueError as e:
        if "collinear" in str(e):
            raised = True
    assert raised, "Expected ValueError('collinear') for collinear ring input"


def test_zero_lad_direction_input():
    """Test 9: Zero-length LAD vector raises ValueError."""
    cor_normal = np.array([0.0, 0.0, 1.0])
    zero_lad = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])

    raised = False
    try:
        compute_iv_normal(cor_normal, zero_lad)
    except ValueError as e:
        if "zero length" in str(e):
            raised = True
    assert raised, "Expected ValueError('zero length') for zero LAD vector"


def test_rca_unresolved_case_handling():
    """Test 10: rca_resolved = False skips alignment with status 'rca_unresolved'."""
    res = process_patient_cardiac_frame(
        centerlines_scanner={},
        landmarks_scanner={},
        patient_id="82.label",
        rca_resolved=False,
        rca_unresolved_reason="RCA stub < 15mm",
    )

    assert res["batch2_status"] == "rca_unresolved"
    assert res["batch2_passed"] is False
    assert "rca unresolved" in res["rejection_reason"].lower()
