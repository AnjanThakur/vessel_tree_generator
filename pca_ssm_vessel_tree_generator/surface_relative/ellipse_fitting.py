"""2D Ellipse Fitting Engine and Exact Coordinate Projection Mathematics for Batch 3.

Implements Part 4 §4.1 of the Technical Design Document:
- compute_ellipse_axis_extents(): Standalone math function computing exact coordinate
  bounding extents (extent_u, extent_v) along canonical axes for a rotated 2D ellipse.
- fit_ellipse_2d(): General 2D EllipseModel fitting.
- fit_coronary_ellipse(): 2D ellipse fit on RCA + LCX ring points in Z=0 plane (a, b_cor).
- fit_iv_ellipse(): 2D ellipse fit on LAD points in X=0 plane (b_IV, c).
"""

from __future__ import annotations

import math
import warnings
from typing import Any
import numpy as np
from skimage.measure import EllipseModel

from surface_relative.fixed_representation import resample_centerline_arc_length

EPS = 1.0e-12


def compute_ellipse_axis_extents(a_raw: float, b_raw: float, theta_rad: float) -> tuple[float, float]:
    """Compute exact mathematical bounding extents (extent_u, extent_v) along coordinate axes.

    For a 2D ellipse rotated by angle theta:
    x(t) = a_raw * cos(t) * cos(theta) - b_raw * sin(t) * sin(theta)
    y(t) = a_raw * cos(t) * sin(theta) + b_raw * sin(t) * cos(theta)

    Evaluating maximum coordinate extents (dx/dt = 0, dy/dt = 0) yields:
    extent_u = sqrt(a_raw^2 * cos^2(theta) + b_raw^2 * sin^2(theta))
    extent_v = sqrt(a_raw^2 * sin^2(theta) + b_raw^2 * cos^2(theta))
    """
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    extent_u = math.sqrt((a_raw * cos_t) ** 2 + (b_raw * sin_t) ** 2)
    extent_v = math.sqrt((a_raw * sin_t) ** 2 + (b_raw * cos_t) ** 2)
    return extent_u, extent_v


def fit_ellipse_2d(points_2d: np.ndarray, min_points: int = 5) -> dict[str, Any]:
    """Fit a general 2D ellipse using skimage.measure.EllipseModel and compute aligned axis extents."""
    pts = np.asarray(points_2d, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points_2d must have shape (N, 2); got {pts.shape}")
    if len(pts) < min_points:
        raise ValueError(f"at least {min_points} points required for 2D ellipse fitting; got {len(pts)}")
    if not np.all(np.isfinite(pts)):
        raise ValueError("non-finite (NaN or Inf) values in 2D point cloud")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = EllipseModel()
        success = model.estimate(pts)
        if not success or getattr(model, "params", None) is None:
            raise ValueError("EllipseModel fitting failed to converge")
        xc, yc, a_raw, b_raw, theta_rad = model.params

    a_raw = float(abs(a_raw))
    b_raw = float(abs(b_raw))
    theta_rad = float(theta_rad) % math.pi

    if a_raw <= EPS or b_raw <= EPS:
        raise ValueError("EllipseModel returned degenerate semi-axis length")

    extent_u, extent_v = compute_ellipse_axis_extents(a_raw, b_raw, theta_rad)

    # Compute RMS point-to-ellipse residual
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        residuals = model.residuals(pts)

    rms_residual = float(np.sqrt(np.mean(residuals**2))) if len(residuals) > 0 else 0.0

    return {
        "center_u_mm": float(xc),
        "center_v_mm": float(yc),
        "a_raw_mm": a_raw,
        "b_raw_mm": b_raw,
        "theta_rad": theta_rad,
        "theta_deg": math.degrees(theta_rad),
        "extent_u_mm": extent_u,
        "extent_v_mm": extent_v,
        "rms_residual_mm": rms_residual,
        "point_count": len(pts),
    }


def fit_coronary_ellipse(rca_cardiac: np.ndarray, lcx_cardiac: np.ndarray) -> dict[str, Any]:
    """Fit Coronary Ellipse on RCA + LCX ring points projected to Z=0 plane.

    Returns raw parameters and aligned semi-axes:
    a = extent_x (cardiac X semi-axis)
    b_cor = extent_y (cardiac Y semi-axis)
    """
    rca_pts = np.asarray(rca_cardiac, dtype=float)
    lcx_pts = np.asarray(lcx_cardiac, dtype=float)

    if rca_pts.ndim != 2 or rca_pts.shape[1] != 3 or lcx_pts.ndim != 2 or lcx_pts.shape[1] != 3:
        raise ValueError("rca_cardiac and lcx_cardiac must have shape (N, 3)")

    rca_resampled = resample_centerline_arc_length(rca_pts, max(len(rca_pts), 15))
    lcx_resampled = resample_centerline_arc_length(lcx_pts, max(len(lcx_pts), 10))

    ring_3d = np.vstack([rca_resampled, lcx_resampled])
    ring_2d = ring_3d[:, :2]  # (x, y) coordinates in Z=0 plane

    fit_res = fit_ellipse_2d(ring_2d)

    return {
        "raw_ellipse": {
            "center_x_mm": fit_res["center_u_mm"],
            "center_y_mm": fit_res["center_v_mm"],
            "a_raw_mm": fit_res["a_raw_mm"],
            "b_raw_mm": fit_res["b_raw_mm"],
            "theta_rad": fit_res["theta_rad"],
            "theta_deg": fit_res["theta_deg"],
            "rms_residual_mm": fit_res["rms_residual_mm"],
        },
        "aligned_axes": {
            "a_mm": fit_res["extent_u_mm"],
            "b_cor_mm": fit_res["extent_v_mm"],
        },
        "point_count": fit_res["point_count"],
    }


def fit_iv_ellipse(lad_cardiac: np.ndarray) -> dict[str, Any]:
    """Fit Interventricular (IV) Ellipse on LAD points projected to X=0 plane.

    Returns raw parameters and aligned semi-axes:
    b_IV = extent_y (cardiac Y semi-axis)
    c = extent_z (cardiac Z semi-axis, base-to-apex)

    Handles degenerate/collinear LAD arcs smoothly via physical bounding box proxy.
    """
    lad_pts = np.asarray(lad_cardiac, dtype=float)
    if lad_pts.ndim != 2 or lad_pts.shape[1] != 3 or len(lad_pts) < 2:
        raise ValueError("lad_cardiac must have shape (N, 3) with N >= 2")

    lad_resampled = resample_centerline_arc_length(lad_pts, max(len(lad_pts), 12))
    lad_2d = lad_resampled[:, 1:3]  # (y, z) coordinates in X=0 plane

    # Physical bounding extent fallback
    y_min, y_max = float(np.min(lad_resampled[:, 1])), float(np.max(lad_resampled[:, 1]))
    z_min, z_max = float(np.min(lad_resampled[:, 2])), float(np.max(lad_resampled[:, 2]))
    phys_b_iv = max(abs(y_max - y_min) / 2.0, 15.0)
    phys_c = max(abs(z_max - z_min), 30.0)

    try:
        fit_res = fit_ellipse_2d(lad_2d)
        c_candidate = fit_res["extent_v_mm"]
        b_iv_candidate = fit_res["extent_u_mm"]

        # Validate against anatomical sanity range [10mm, 200mm]
        if 10.0 <= c_candidate <= 200.0 and 10.0 <= b_iv_candidate <= 200.0:
            return {
                "raw_ellipse": {
                    "center_y_mm": fit_res["center_u_mm"],
                    "center_z_mm": fit_res["center_v_mm"],
                    "b_iv_raw_mm": fit_res["a_raw_mm"],
                    "c_raw_mm": fit_res["b_raw_mm"],
                    "theta_iv_rad": fit_res["theta_rad"],
                    "theta_iv_deg": fit_res["theta_deg"],
                    "rms_residual_mm": fit_res["rms_residual_mm"],
                    "fit_source": "skimage_ellipse_fit",
                },
                "aligned_axes": {
                    "b_iv_mm": b_iv_candidate,
                    "c_mm": c_candidate,
                },
                "point_count": fit_res["point_count"],
            }
    except Exception:
        pass

    # Fallback to physical LAD descent bounding extent
    return {
        "raw_ellipse": {
            "center_y_mm": (y_min + y_max) / 2.0,
            "center_z_mm": (z_min + z_max) / 2.0,
            "b_iv_raw_mm": phys_b_iv,
            "c_raw_mm": phys_c,
            "theta_iv_rad": 0.0,
            "theta_iv_deg": 0.0,
            "rms_residual_mm": 0.0,
            "fit_source": "physical_lad_descent_span_proxy",
        },
        "aligned_axes": {
            "b_iv_mm": phys_b_iv,
            "c_mm": phys_c,
        },
        "point_count": len(lad_resampled),
    }
