"""Common rigid cardiac frame construction and individual metrics validation."""

from __future__ import annotations

import math
from typing import Any
import numpy as np

EPS = 1.0e-12


def unit(vector: np.ndarray, name: str = "vector") -> np.ndarray:
    """Normalize a 3D vector."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError(f"{name} has zero or non-finite length")
    return vector / norm


def compute_cardiac_frame(
    coronary_normal: np.ndarray,
    iv_normal: np.ndarray,
    coronary_centroid: np.ndarray,
    lad_centerline: np.ndarray,
    lca_ostium: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct the rigid cardiac frame transformation (Origin, R_total) per patient.

    Convention (Design Doc §2.3 & §3.3):
    - z_axis: coronary plane normal, oriented so LAD descends in -Z (lad_vector . z < 0).
    - x_axis: IV plane normal (left <-> right).
    - y_axis: unit(z_axis x x_axis) (anterior <-> posterior).
    - re-derived x_axis: unit(y_axis x z_axis).
    - origin: coronary plane centroid.
    - Canonical Z-rotation: rotate around z by delta_theta so LCA ostium sits at +X (angle = 0).

    Returns:
        origin: (3,) float array
        R_total: (3, 3) rotation matrix mapping scanner RAS -> global cardiac frame
        metadata: dict containing intermediate frame parameters
    """
    centroid = np.asarray(coronary_centroid, dtype=float).reshape(3)
    axis_z = unit(coronary_normal, "coronary_normal")
    axis_x = unit(iv_normal, "iv_normal")

    # Orient z_axis so LAD descends toward negative z
    lad_centerline = np.asarray(lad_centerline, dtype=float)
    if len(lad_centerline) >= 2:
        lad_dir = lad_centerline[-1] - lad_centerline[0]
        if np.dot(lad_dir, axis_z) > 0:
            axis_z = -axis_z

    axis_y = np.cross(axis_z, axis_x)
    axis_y_norm = np.linalg.norm(axis_y)
    if axis_y_norm <= EPS:
        raise ValueError("coronary_normal and iv_normal are collinear")
    axis_y /= axis_y_norm

    axis_x = np.cross(axis_y, axis_z)
    axis_x /= np.linalg.norm(axis_x)

    # Initial frame rotation (scanner -> initial cardiac frame)
    R_initial = np.stack([axis_x, axis_y, axis_z], axis=0)  # 3x3

    # Project LCA ostium into initial cardiac frame to canonicalize Z-rotation
    lca_ostium_cardiac_init = (np.asarray(lca_ostium, dtype=float).reshape(3) - centroid) @ R_initial.T
    lca_ost_xy = lca_ostium_cardiac_init[:2]
    lca_angle = math.atan2(lca_ost_xy[1], lca_ost_xy[0])
    delta_theta = -lca_angle  # Rotate so LCA ostium sits at angle = 0 (+X axis)

    R_z = np.array(
        [
            [math.cos(delta_theta), -math.sin(delta_theta), 0.0],
            [math.sin(delta_theta), math.cos(delta_theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    # Total rotation matrix: R_total = R_z @ R_initial
    R_total = R_z @ R_initial

    metadata = {
        "origin": centroid,
        "axis_x": R_total[0],
        "axis_y": R_total[1],
        "axis_z": R_total[2],
        "delta_theta_rad": delta_theta,
        "lca_ostium_initial_angle_deg": math.degrees(lca_angle),
    }

    return centroid, R_total, metadata


def transform_to_cardiac_frame(points: np.ndarray, origin: np.ndarray, R_total: np.ndarray) -> np.ndarray:
    """Transform 3D point cloud from scanner RAS into global cardiac frame (rigidly)."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 1 and len(points) == 3:
        return (points - origin) @ R_total.T
    return (points - origin) @ R_total.T


def validate_cardiac_frame(
    origin: np.ndarray,
    R_total: np.ndarray,
    branches_raw: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Perform strict 8-point individual validation check on patient cardiac frame transformation.

    Checks:
    1. right_handed: det(R_total) > 0
    2. determinant: float(det(R_total)) (approx +1.0)
    3. orthogonality_error: max(abs(R_total @ R_total.T - I))
    4. axis_norm_error: max(abs(norm(axis) - 1.0)) for each row
    5. length_error per branch: max segment length change before vs after transform (must be <= 1e-6 mm)
    6. LAD_apex_alignment: dot product of LAD descent vector with z_axis in local frame (must be < 0)
    7. reflection_detected: det(R_total) <= 0
    8. overall_pass: All individual criteria satisfied
    """
    det_val = float(np.linalg.det(R_total))
    right_handed = det_val > 0.0
    reflection_detected = det_val <= 0.0 or abs(det_val - 1.0) > 1.0e-3

    ortho_matrix = R_total @ R_total.T
    ortho_error = float(np.max(np.abs(ortho_matrix - np.eye(3))))

    axis_norms = [float(np.linalg.norm(row)) for row in R_total]
    axis_norm_error = float(np.max(np.abs(np.array(axis_norms) - 1.0)))

    length_errors = {}
    for name, raw_pts in branches_raw.items():
        if raw_pts is not None and len(raw_pts) >= 2:
            cardiac_pts = transform_to_cardiac_frame(raw_pts, origin, R_total)
            before_lengths = np.linalg.norm(np.diff(raw_pts, axis=0), axis=1)
            after_lengths = np.linalg.norm(np.diff(cardiac_pts, axis=0), axis=1)
            length_errors[f"length_error_{name}"] = float(np.max(np.abs(before_lengths - after_lengths)))
        else:
            length_errors[f"length_error_{name}"] = 0.0

    # LAD apex alignment check
    lad_raw = branches_raw.get("lad")
    if lad_raw is not None and len(lad_raw) >= 2:
        lad_cardiac = transform_to_cardiac_frame(lad_raw, origin, R_total)
        lad_descent_z = float(lad_cardiac[-1][2] - lad_cardiac[0][2])
        lad_apex_aligned = lad_descent_z < 0.0
    else:
        lad_descent_z = 0.0
        lad_apex_aligned = False

    max_length_err = max(length_errors.values()) if length_errors else 0.0
    overall_pass = (
        right_handed
        and not reflection_detected
        and ortho_error < 1.0e-6
        and axis_norm_error < 1.0e-6
        and max_length_err < 1.0e-5
        and lad_apex_aligned
    )

    res = {
        "right_handed": right_handed,
        "determinant": det_val,
        "orthogonality_error": ortho_error,
        "axis_norm_error": axis_norm_error,
        "LAD_apex_alignment_z_diff": lad_descent_z,
        "LAD_apex_aligned": lad_apex_aligned,
        "reflection_detected": reflection_detected,
        "overall_pass": overall_pass,
    }
    res.update(length_errors)
    return res
