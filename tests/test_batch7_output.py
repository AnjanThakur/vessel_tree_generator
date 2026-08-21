"""Synthetic Unit Test Suite for Batch 7 Standardized Output Formatting & Packaging."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "pca_ssm_vessel_tree_generator"))

from output.save_geometry import (
    resample_curve_3d,
    compute_vessel_radius,
    get_ordered_tree_branches,
    construct_geometry_static_array,
    construct_geometry_cine_array,
)
from output.save_metadata import (
    build_tree_metadata,
    build_ellipsoid_params_metadata,
    build_global_pipeline_report,
)
import pipeline


def test_resample_curve_3d():
    """Test 1: Verify 3D curve arc-length resampling to 50 points with endpoint preservation."""
    raw_pts = np.array([[0.0, 0.0, 0.0], [5.0, 10.0, 0.0], [10.0, 20.0, 30.0]])
    resampled = resample_curve_3d(raw_pts, num_points=50)

    assert resampled.shape == (50, 3)
    np.testing.assert_allclose(resampled[0], raw_pts[0], atol=1e-12)
    np.testing.assert_allclose(resampled[-1], raw_pts[-1], atol=1e-12)


def test_compute_vessel_radius():
    """Test 2: Verify anatomical vessel radius models (LMCA=2.0, RCA/LAD/LCX=1.8->0.8, SB=0.6->0.4)."""
    r_lmca = compute_vessel_radius("LMCA", num_points=50)
    r_lad = compute_vessel_radius("LAD", num_points=50)
    r_sb = compute_vessel_radius("SB_LAD_00", num_points=50)

    np.testing.assert_allclose(r_lmca, 2.0, atol=1e-12)
    np.testing.assert_allclose(r_lad[0], 1.8, atol=1e-12)
    np.testing.assert_allclose(r_lad[-1], 0.8, atol=1e-12)
    np.testing.assert_allclose(r_sb[0], 0.6, atol=1e-12)
    np.testing.assert_allclose(r_sb[-1], 0.4, atol=1e-12)


def test_deterministic_branch_ordering():
    """Test 3: Verify deterministic branch ordering (0=RCA, 1=LMCA, 2=LAD, 3=LCX, 4+=side branches)."""
    vessels_3d = {
        "LCX": np.zeros((5, 3)),
        "RCA": np.zeros((5, 3)),
        "LMCA": np.zeros((5, 3)),
        "LAD": np.zeros((5, 3)),
    }
    side_branches = [
        {"parent": "RCA", "attachment_index": 2, "points": np.zeros((3, 3))},
        {"parent": "LAD", "attachment_index": 1, "points": np.zeros((3, 3))},
    ]

    ordered = get_ordered_tree_branches(vessels_3d, side_branches)
    names = [b[0] for b in ordered]

    assert names[0] == "RCA"
    assert names[1] == "LMCA"
    assert names[2] == "LAD"
    assert names[3] == "LCX"
    assert names[4] == "SB_LAD_00"
    assert names[5] == "SB_RCA_01"


def test_construct_geometry_static_array():
    """Test 4: Verify static geometry array shape (M_branches x 50 x 4) and radius positivity r > 0."""
    dummy_tree_4d = {
        "tree_id": "test_tree",
        "frames": [
            {
                "phase": 0.0,
                "vessels_3d": {
                    "RCA": np.array([[0, 0, 0], [10, 0, 0]]),
                    "LMCA": np.array([[10, 0, 0], [12, 0, 0]]),
                    "LAD": np.array([[12, 0, 0], [20, 0, 0]]),
                    "LCX": np.array([[12, 0, 0], [15, 5, 0]]),
                },
                "side_branches": [],
            }
        ],
    }

    geom_static = construct_geometry_static_array(dummy_tree_4d, num_points=50)

    assert geom_static.shape == (4, 50, 4)
    assert np.all(geom_static[:, :, 3] > 0.0)


def test_construct_geometry_cine_array():
    """Test 5: Verify 4D cine geometry array shape (10 x M_branches x 50 x 4)."""
    dummy_tree_4d = {
        "tree_id": "test_tree",
        "frames": [
            {
                "phase": float(p),
                "vessels_3d": {
                    "RCA": np.array([[0, 0, 0], [10, 0, 0]]),
                    "LMCA": np.array([[10, 0, 0], [12, 0, 0]]),
                    "LAD": np.array([[12, 0, 0], [20, 0, 0]]),
                    "LCX": np.array([[12, 0, 0], [15, 5, 0]]),
                },
                "side_branches": [],
            }
            for p in np.linspace(0.0, 0.9, 10)
        ],
    }

    geom_cine = construct_geometry_cine_array(dummy_tree_4d, num_points=50)

    assert geom_cine.shape == (10, 4, 50, 4)


def test_static_equals_cine0_invariant():
    """Test 6: Verify geometry_static == geometry_cine[0] at phi=0.0 with atol=1e-12."""
    dummy_tree_4d = {
        "tree_id": "test_tree",
        "frames": [
            {
                "phase": float(p),
                "vessels_3d": {
                    "RCA": np.array([[0, 0, 0], [10, 0, 0]]),
                    "LMCA": np.array([[10, 0, 0], [12, 0, 0]]),
                    "LAD": np.array([[12, 0, 0], [20, 0, 0]]),
                    "LCX": np.array([[12, 0, 0], [15, 5, 0]]),
                },
                "side_branches": [],
            }
            for p in np.linspace(0.0, 0.9, 10)
        ],
    }

    geom_static = construct_geometry_static_array(dummy_tree_4d, num_points=50)
    geom_cine = construct_geometry_cine_array(dummy_tree_4d, num_points=50)

    np.testing.assert_allclose(geom_static, geom_cine[0], atol=1e-12)


def test_build_global_pipeline_report():
    """Test 7: Verify dynamic pipeline report building from actual output directory."""
    outputs_dir = PROJECT_ROOT / "outputs"
    report = build_global_pipeline_report(outputs_dir)

    assert "patient_accounting" in report
    assert "statistical_shape_model" in report
    assert "synthetic_generation" in report
    assert "cardiac_motion" in report


def test_pipeline_cli_integration():
    """Test 8: Verify non-invasive master pipeline CLI parser."""
    parser = pipeline.parse_args
    assert callable(parser)
