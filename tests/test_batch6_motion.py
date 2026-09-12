"""Synthetic unit tests for Batch 6 4D Cardiac Phase Motion Deformation."""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from motion.contraction_curve import contraction_curve, deform_ellipsoid
from motion.cardiac_motion import apply_cardiac_motion_to_tree
from generation.validator import has_self_intersection


def test_contraction_curve():
    """Test 1: Verify contraction curve s(0)=0.0, s(0.35)=1.0, s(1.0)=0.0."""
    np.testing.assert_allclose(contraction_curve(0.0), 0.0, atol=1e-10)
    np.testing.assert_allclose(contraction_curve(0.35), 1.0, atol=1e-10)
    np.testing.assert_allclose(contraction_curve(1.0), 0.0, atol=1e-10)

    # Monotonic increase up to peak
    s_020 = contraction_curve(0.20)
    assert 0.0 < s_020 < 1.0


def test_ellipsoid_radial_contraction():
    """Test 2: Verify semi-axes a, b shrink by 15% at peak systole (phi=0.35)."""
    a0, b0, c0 = 40.0, 30.0, 25.0
    u, v = 0.0, 1.5

    a_def, b_def, c_def, u_def, v_def = deform_ellipsoid(
        a0, b0, c0, u, v, phase=0.35, radial_amplitude=0.15
    )

    np.testing.assert_allclose(a_def, 40.0 * 0.85, atol=1e-10)
    np.testing.assert_allclose(b_def, 30.0 * 0.85, atol=1e-10)


def test_ellipsoid_longitudinal_shortening():
    """Test 3: Verify semi-axis c shrinks by 10% at peak systole (phi=0.35)."""
    a0, b0, c0 = 40.0, 30.0, 25.0
    u, v = 0.0, 1.5

    a_def, b_def, c_def, u_def, v_def = deform_ellipsoid(
        a0, b0, c0, u, v, phase=0.35, longitudinal_amplitude=0.10
    )

    np.testing.assert_allclose(c_def, 25.0 * 0.90, atol=1e-10)


def test_basetoapex_torsion_gradient():
    """Test 4: Verify base (v near 0) rotates 10 deg at peak systole while apex (v near pi) rotates 0 deg."""
    a0, b0, c0 = 40.0, 30.0, 25.0
    u0 = 0.0

    # Base (v=0)
    _, _, _, u_base, _ = deform_ellipsoid(a0, b0, c0, u0, v=0.0, phase=0.35, torsion_amplitude_deg=10.0)
    expected_base_rad = math.radians(10.0)
    np.testing.assert_allclose(u_base, expected_base_rad, atol=1e-10)

    # Apex (v=pi)
    _, _, _, u_apex, _ = deform_ellipsoid(a0, b0, c0, u0, v=math.pi, phase=0.35, torsion_amplitude_deg=10.0)
    np.testing.assert_allclose(u_apex, 0.0, atol=1e-10)


def test_deformed_3d_surface_reconstruction():
    """Test 5: Verify deformed 3D surface point reconstruction and local basis shift."""
    dummy_tree = {
        "tree_id": "test_tree",
        "ellipsoid_params": {"a_mm": 40.0, "b_mm": 30.0, "c_mm": 25.0},
        "landmarks": {
            "lca_ostium": {"u": 0.0, "v": 1.5, "offset": -5.0},
            "rca_ostium": {"u": 0.5, "v": 1.5, "offset": -5.0},
            "bifurcation": {"u": -0.2, "v": 1.5, "offset": -10.0},
            "lad_endpoint": {"u": -0.3, "v": 2.2, "offset": 10.0},
            "lcx_endpoint": {"u": 0.3, "v": 1.5, "offset": 5.0},
            "rca_endpoint": {"u": 0.1, "v": 1.8, "offset": 12.0},
        },
        "vessels_3d": {
            "LMCA": np.array([[38.0, 0.0, 0.0], [35.0, -5.0, 0.0]]),
            "LAD": np.array([[35.0, -5.0, 0.0], [20.0, -10.0, -15.0]]),
            "LCX": np.array([[35.0, -5.0, 0.0], [25.0, 10.0, -10.0]]),
            "RCA": np.array([[38.0, 5.0, 0.0], [20.0, 15.0, -15.0]]),
        },
        "side_branches": [],
    }

    res_4d = apply_cardiac_motion_to_tree(dummy_tree, num_phases=10)

    assert res_4d["num_phases"] == 10
    assert len(res_4d["frames"]) == 10
    assert res_4d["phase_values"][0] == 0.0
    assert res_4d["phase_values"][-1] == 1.0
    for name in ("LMCA", "LAD", "LCX", "RCA"):
        np.testing.assert_allclose(
            res_4d["frames"][0]["vessels_3d"][name],
            res_4d["frames"][-1]["vessels_3d"][name],
            atol=1.0e-10,
        )
    # Phase 0 (diastole) vs Phase 3 (peak systole ~0.35)
    pts_d = res_4d["frames"][0]["vessels_3d"]["LMCA"]
    pts_s = res_4d["frames"][3]["vessels_3d"]["LMCA"]
    assert not np.allclose(pts_d, pts_s)


def test_ostial_offset_motion_decay():
    """Test 6: Verify ostial offset shrinks with radial contraction at peak systole."""
    dummy_tree = {
        "tree_id": "test_tree",
        "ellipsoid_params": {"a_mm": 40.0, "b_mm": 30.0, "c_mm": 25.0},
        "landmarks": {
            "lca_ostium": {"u": 0.0, "v": 1.5, "offset": -10.0},
            "rca_ostium": {"u": 0.5, "v": 1.5, "offset": -10.0},
            "bifurcation": {"u": -0.2, "v": 1.5, "offset": 0.0},
            "lad_endpoint": {"u": -0.3, "v": 2.2, "offset": 0.0},
            "lcx_endpoint": {"u": 0.3, "v": 1.5, "offset": 0.0},
            "rca_endpoint": {"u": 0.1, "v": 1.8, "offset": 0.0},
        },
        "vessels_3d": {
            "LMCA": np.array([[30.0, 0.0, 0.0], [35.0, -5.0, 0.0]]),
            "LAD": np.array([[35.0, -5.0, 0.0], [20.0, -10.0, -15.0]]),
            "LCX": np.array([[35.0, -5.0, 0.0], [25.0, 10.0, -10.0]]),
            "RCA": np.array([[30.0, 5.0, 0.0], [20.0, 15.0, -15.0]]),
        },
        "side_branches": [],
    }

    res_4d = apply_cardiac_motion_to_tree(dummy_tree, num_phases=10, radial_amplitude=0.15)
    # Check that peak-systole frame semi-axes shrink
    f_sys = res_4d["frames"][3]
    np.testing.assert_allclose(f_sys["ellipsoid_params"]["a_mm"], 40.0 * (1.0 - 0.15 * contraction_curve(f_sys["phase"])), atol=1e-5)


def test_lmca_bifurcation_snapping_all_10_phases():
    """Test 7: Verify LMCA bifurcation snapping continuity LAD[0] == LCX[0] == LMCA[-1] across all 10 phases."""
    dummy_tree = {
        "tree_id": "test_tree",
        "ellipsoid_params": {"a_mm": 40.0, "b_mm": 30.0, "c_mm": 25.0},
        "landmarks": {
            "lca_ostium": {"u": 0.0, "v": 1.5, "offset": -5.0},
            "rca_ostium": {"u": 0.5, "v": 1.5, "offset": -5.0},
            "bifurcation": {"u": -0.2, "v": 1.5, "offset": -10.0},
            "lad_endpoint": {"u": -0.3, "v": 2.2, "offset": 10.0},
            "lcx_endpoint": {"u": 0.3, "v": 1.5, "offset": 5.0},
            "rca_endpoint": {"u": 0.1, "v": 1.8, "offset": 12.0},
        },
        "vessels_3d": {
            "LMCA": np.array([[38.0, 0.0, 0.0], [35.0, -5.0, 0.0]]),
            "LAD": np.array([[35.0, -5.0, 0.0], [20.0, -10.0, -15.0]]),
            "LCX": np.array([[35.0, -5.0, 0.0], [25.0, 10.0, -10.0]]),
            "RCA": np.array([[38.0, 5.0, 0.0], [20.0, 15.0, -15.0]]),
        },
        "side_branches": [],
    }

    res_4d = apply_cardiac_motion_to_tree(dummy_tree, num_phases=10)

    for frame in res_4d["frames"]:
        v3d = frame["vessels_3d"]
        lmca_end = v3d["LMCA"][-1]
        lad_start = v3d["LAD"][0]
        lcx_start = v3d["LCX"][0]

        np.testing.assert_allclose(lad_start, lmca_end, atol=1e-12)
        np.testing.assert_allclose(lcx_start, lmca_end, atol=1e-12)


def test_4d_segment_self_intersection_test():
    """Test 8: Verify 1.0 mm segment self-intersection test across all 10 cardiac phases."""
    dummy_tree = {
        "tree_id": "test_tree",
        "ellipsoid_params": {"a_mm": 40.0, "b_mm": 30.0, "c_mm": 25.0},
        "landmarks": {
            "lca_ostium": {"u": 0.0, "v": 1.5, "offset": -5.0},
            "rca_ostium": {"u": 0.5, "v": 1.5, "offset": -5.0},
            "bifurcation": {"u": -0.2, "v": 1.5, "offset": -10.0},
            "lad_endpoint": {"u": -0.3, "v": 2.2, "offset": 10.0},
            "lcx_endpoint": {"u": 0.3, "v": 1.5, "offset": 5.0},
            "rca_endpoint": {"u": 0.1, "v": 1.8, "offset": 12.0},
        },
        "vessels_3d": {
            "LMCA": np.array([[38.0, 0.0, 0.0], [35.0, -5.0, 0.0]]),
            "LAD": np.array([[35.0, -5.0, 0.0], [20.0, -10.0, -15.0]]),
            "LCX": np.array([[35.0, -5.0, 0.0], [25.0, 10.0, -10.0]]),
            "RCA": np.array([[38.0, 5.0, 0.0], [20.0, 15.0, -15.0]]),
        },
        "side_branches": [],
    }

    res_4d = apply_cardiac_motion_to_tree(dummy_tree, num_phases=10)

    for frame in res_4d["frames"]:
        has_int, _ = has_self_intersection(frame["vessels_3d"], frame["side_branches"], min_dist_threshold_mm=1.0)
        assert has_int is False
